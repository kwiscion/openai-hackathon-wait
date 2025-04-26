import asyncio
import json
import sys
from typing import Dict

import dotenv

# Import Agent/Runner from the SDK
from agents import Runner
from loguru import logger
from openai import AsyncOpenAI

from openai_hackathon_wait.agents.reviewer import (
    ReviewerContext,
    create_reviewer_agent,
)
from openai_hackathon_wait.agents.structure_validator import run_validator_agent

# Import agent creation functions and models directly
from .agents.reviewer_finder import (
    ProposedReviewers,  # Needed for type checking/casting
    ReviewerAssessment,  # Needed for type checking/casting
    create_reviewer_proposer_agent,
    create_reviewer_selector_agent,
)
from .publication_decision import (
    PublicationDecisionContext,
    create_publication_decision_agent,
)
from .review_synthesizer import create_synthesizer_agent

dotenv.load_dotenv()


async def main(paper_path: str):
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
    # rag, paper_context = await create_context(client, paper_id, paper_content)

    # Run the structure validator
    structure_validator_result = await run_validator_agent(
        paper_content=paper_content, auto_detect=True, grammar_check=True
    )

    #############################
    # Reviewer Agents Selection #
    #############################
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

        logger.info(
            f"Found {len(selected_reviewers_dict)} reviewers: {list(selected_reviewers_dict.keys())}"
        )
    except Exception as e:
        logger.error(f"Error during reviewer finding process: {e}")
        # Decide how to handle failure (exit, continue with defaults?)
        sys.exit(1)  # Exit for now

    ############################
    #      Reviewer Agents     #
    ############################
    # Run the review for each selected reviewer
    try:
        review_jobs = []
        logger.info(
            f"Starting review process with {len(selected_reviewers_dict)} selected reviewers..."
        )
        for reviewer_name, system_prompt in selected_reviewers_dict.items():
            logger.info(f"Initializing reviewer: {reviewer_name}")
            reviewer = create_reviewer_agent(name=reviewer_name)
            context = ReviewerContext(
                reviewer_persona=system_prompt,
                paper_content=paper_content,
                literature_context="",
                technical_context=structure_validator_result,
                vector_store_name="",
            )
            review_jobs.append(
                Runner.run(reviewer, "Review the paper", context=context)
            )

        reviews = await asyncio.gather(*review_jobs)
    except Exception as e:
        logger.error(f"Error during review process: {e}")
        sys.exit(1)

    ############################
    # Review Synthesizer Agent #
    ############################
    # Synthesize the reviews
    try:
        synthesizer = create_synthesizer_agent()
        reviews_dict = [review.final_output.model_dump() for review in reviews]
        reviews_str = json.dumps(reviews_dict)
        synthesis = await Runner.run(synthesizer, reviews_str)
    except Exception as e:
        logger.error(f"Error during review synthesis: {e}")
        sys.exit(1)

    ##############################
    # Publication Decision Agent #
    ##############################
    try:
        decision_agent = create_publication_decision_agent()
        context = PublicationDecisionContext(
            synthesized_review=synthesis.final_output.model_dump(),
            manuscript=paper_content,
            manuscript_filename=paper_path,
        )
        decision = await Runner.run(
            decision_agent, "Decide if paper is ready for publication", context=context
        )
    except Exception as e:
        logger.error(f"Error during publication decision: {e}")
        sys.exit(1)

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    try:
        with open(decision_output_path, "w", encoding="utf-8") as f:
            json.dump(decision.final_output.model_dump(), f, indent=4)
        logger.info(f"Decision saved to {decision_output_path}")
    except Exception as e:
        logger.error(f"Error saving decision to {decision_output_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openai_hackathon_wait.main <paper_path>")
        sys.exit(1)

    paper_path = sys.argv[1]
    asyncio.run(main(paper_path))
