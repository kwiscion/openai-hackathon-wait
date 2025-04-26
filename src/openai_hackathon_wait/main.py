import asyncio
import json
import sys
from typing import Dict

import dotenv

# Import Agent/Runner from the SDK
from agents import Runner
from loguru import logger
from openai import AsyncOpenAI

from openai_hackathon_wait.agents.reviewer import create_reviewer_agent

# Import agent creation functions and models directly
from .agents.reviewer_finder import (
    ProposedReviewers,  # Needed for type checking/casting
    ReviewerAssessment,  # Needed for type checking/casting
    create_reviewer_proposer_agent,
    create_reviewer_selector_agent,
)
from .create_context import create_context
from .publication_decision import PublicationDecisionOrchestrator
from .review_synthesizer import create_synthesizer_agent

dotenv.load_dotenv()


async def main(paper_path: str, num_reviews: int):
    """
    Main method to run the review orchestrator.

    Args:
        paper_path: Path to the paper file
        num_reviews: Target number of reviews (currently unused as reviewer count is determined by selection)
    """
    client = AsyncOpenAI()
    # Read the paper
    try:
        with open(paper_path, "r", encoding="utf-8") as f:
            paper_content = f.read()
    except FileNotFoundError:
        logger.error(f"Paper file not found: {paper_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading paper file {paper_path}: {e}")
        sys.exit(1)

    logger.info(f"Paper: {paper_content[:100]}...")
    paper_id = "paper_" + paper_path.split("/")[-1].split(".")[0][:50]

    # Create the context
    paper_context = await create_context(client, paper_id, paper_content)

    # Run the structure validator
    structure_validator_result = await run_validator(
        paper_content=paper_content, auto_detect=True, grammar_check=True
    )

    additional_analysis = [
        {"area": "structure and language", "review": structure_validator_result}
    ]

    # --- Find Reviewers using direct agent calls ---
    selected_reviewers_dict: Dict[str, str] | None = None
    try:
        # 1. Create and run Proposer Agent
        logger.info("Finding reviewers: Running proposer agent...")
        proposer_agent = create_reviewer_proposer_agent()
        proposed_result = await Runner.run(proposer_agent, paper_content)
        proposed_reviewers = proposed_result.final_output_as(ProposedReviewers)
        logger.info(f"Proposed {len(proposed_reviewers.reviewers)} reviewers.")

        # 2. Create and run Selector Agent
        logger.info("Finding reviewers: Running selector agent...")
        selector_agent = create_reviewer_selector_agent()
        selection_input = proposed_reviewers.model_dump_json()
        selection_result = await Runner.run(selector_agent, selection_input)
        reviewer_selection = selection_result.final_output_as(ReviewerAssessment)
        logger.info(
            f"Selected {len(reviewer_selection.selected_reviewers)} reviewers based on rationale: {reviewer_selection.selection_rationale}"
        )

        # 3. Extract final dictionary
        selected_reviewers_dict = {
            reviewer.name: reviewer.system_prompt
            for reviewer in reviewer_selection.selected_reviewers
        }

    except Exception as e:
        logger.error(f"Error during reviewer finding process: {e}")
        # Decide how to handle failure (exit, continue with defaults?)
        sys.exit(1)  # Exit for now

    # Check if reviewers were found
    if not selected_reviewers_dict:
        logger.error("Could not find/select reviewers. Exiting.")
        sys.exit(1)

    logger.info(
        f"Found {len(selected_reviewers_dict)} reviewers: {list(selected_reviewers_dict.keys())}"
    )
    # Removed finder.save_reviewers() call
    # --- End Find Reviewers ---

    # Run the review for each selected reviewer
    review_jobs = []
    logger.info(
        f"Starting review process with {len(selected_reviewers_dict)} selected reviewers..."
    )
    for reviewer_name, system_prompt in selected_reviewers_dict.items():
        logger.info(f"Initializing reviewer: {reviewer_name}")
        reviewer = create_reviewer_agent(name=reviewer_name, prompt=system_prompt)
        review_jobs.append(Runner.run(reviewer, paper_content))

    reviews = await asyncio.gather(*review_jobs)

    # Synthesize the reviews
    synthesizer = create_synthesizer_agent()
    reviews_dict = [review.final_output.model_dump() for review in reviews]
    synthesis = await Runner.run(synthesizer, reviews_dict)

    # Save the synthesis
    synthesis_output_path = paper_path.replace(".md", "_synthesis.json")
    try:
        with open(synthesis_output_path, "w", encoding="utf-8") as f:
            json.dump(synthesis.model_dump(), f, indent=4)
        logger.info(f"Synthesis saved to {synthesis_output_path}")
    except Exception as e:
        logger.error(f"Error saving synthesis to {synthesis_output_path}: {e}")

    # Make the decision
    # Pass client to PublicationDecisionOrchestrator if needed
    decision_orchestrator = PublicationDecisionOrchestrator(
        synthesis_output_path,
        paper_path,
        # literature_context_path=context_path,
        # technical_analysis_path=structure_validator_path,
    )

    decision = await decision_orchestrator.make_decision()

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    try:
        with open(decision_output_path, "w", encoding="utf-8") as f:
            json.dump(decision.model_dump(), f, indent=4)
        logger.info(f"Decision saved to {decision_output_path}")
    except Exception as e:
        logger.error(f"Error saving decision to {decision_output_path}: {e}")


if __name__ == "__main__":
    # Keep num_reviews for now, although it's not directly used to set the number of reviewers
    if len(sys.argv) < 3:
        print(
            "Usage: python -m openai_hackathon_wait.main <paper_path> <num_reviews (Note: actual number depends on reviewer selection)>"
        )
        sys.exit(1)

    paper_path = sys.argv[1]
    num_reviews = int(sys.argv[2])  # Not directly used, but kept for consistency
    asyncio.run(main(paper_path, num_reviews))
