import asyncio
import json
import sys

import dotenv

# Import Agent/Runner from the SDK
from loguru import logger
from openai import AsyncOpenAI

from openai_hackathon_wait.agents.reviewer import (
    run_reviewer_agent,
)
from openai_hackathon_wait.agents.reviewer_finder import run_reviewer_finder_agent
from openai_hackathon_wait.agents.structure_validator import run_validator_agent

from .agents.publication_decision import run_publication_decision_agent
from .agents.review_synthesizer import run_synthesizer_agent

# Import agent creation functions and models directly

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

    # Run the reviewer finder agent
    selected_reviewers_dict = await run_reviewer_finder_agent(
        paper_content=paper_content, model="gpt-4o-mini"
    )

    # Run the review for each selected reviewer
    try:
        review_jobs = []
        logger.info(
            f"Starting review process with {len(selected_reviewers_dict)} selected reviewers..."
        )
        for reviewer_name, system_prompt in selected_reviewers_dict.items():
            review_jobs.append(
                run_reviewer_agent(
                    paper_content=paper_content,
                    literature_context="",
                    technical_context=structure_validator_result,
                    reviewer_persona=system_prompt,
                    name=reviewer_name,
                )
            )

        reviews = await asyncio.gather(*review_jobs)
        logger.info("Reviews gathered.")
    except Exception as e:
        logger.error(f"Error during review process: {e}")
        sys.exit(1)

    # Synthesize the reviews
    synthesized_review = await run_synthesizer_agent(reviews)

    # Run the publication decision agent
    decision = await run_publication_decision_agent(
        synthesized_review=synthesized_review,
        manuscript=paper_content,
        manuscript_filename=paper_path,
    )

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    try:
        with open(decision_output_path, "w", encoding="utf-8") as f:
            json.dump(decision.model_dump(), f, indent=4)
        logger.info(f"Decision saved to {decision_output_path}")
    except Exception as e:
        logger.error(f"Error saving decision to {decision_output_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openai_hackathon_wait.main <paper_path>")
        sys.exit(1)

    paper_path = sys.argv[1]
    asyncio.run(main(paper_path))
