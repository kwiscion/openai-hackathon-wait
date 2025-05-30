import asyncio
import sys
from typing import List

import dotenv
from agents import trace
from loguru import logger
from openai import AsyncOpenAI

from openai_hackathon_wait.agents.create_context import create_context
from openai_hackathon_wait.agents.publication_decision import (
    PublicationDecision,
    run_publication_decision_agent,
)
from openai_hackathon_wait.agents.review_synthesizer import run_synthesizer_agent
from openai_hackathon_wait.agents.reviewer import Review, run_reviewer_agent
from openai_hackathon_wait.agents.reviewer_finder import run_reviewer_finder_agent
from openai_hackathon_wait.agents.structure_validator import run_validator_agent

# Import agent creation functions and models directly

dotenv.load_dotenv()


async def review_orchestrator(
    paper_content: str, paper_id: str = "abc"
) -> tuple[PublicationDecision, List[Review]]:
    """
    Main method to run the review orchestrator.

    Args:
        paper_content: The manuscript to review
    """

    with trace("Paper Review"):
        # Initialize the OpenAI client
        client = AsyncOpenAI()

        logger.info(f"Paper: {paper_content[:100]}...")

        # Create the context
        paper_context = await create_context(client, paper_content)

        # Run the structure validator
        structure_validator_result = await run_validator_agent(
            paper_content=paper_content,
            auto_detect=True,
            grammar_check=True,
            model="gpt-4.1",
        )

        # Run the reviewer finder agent
        selected_reviewers_dict = await run_reviewer_finder_agent(
            paper_content=paper_content, model="o3"
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
                        literature_context=paper_context,
                        technical_context=structure_validator_result,
                        reviewer_persona=system_prompt,
                        name=reviewer_name,
                        model="o3",
                    )
                )

            reviews = await asyncio.gather(*review_jobs)
            logger.info("Reviews gathered.")
        except Exception as e:
            logger.error(f"Error during review process: {e}")
            sys.exit(1)

        # Synthesize the reviews
        synthesized_review = await run_synthesizer_agent(reviews, model="gpt-4.1")

        # Run the publication decision agent
        decision = await run_publication_decision_agent(
            synthesized_review=synthesized_review,
            manuscript=paper_content,
            literature_context=paper_context,
            technical_analysis=structure_validator_result,
            model="o3",
        )

        return decision, reviews
