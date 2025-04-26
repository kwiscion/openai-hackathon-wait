import asyncio
import json
import sys
from typing import List

import dotenv
from agents import Runner
from loguru import logger

# Import the main reviewer agent which now has the tool
from .agents.reviewer import Review, reviewer_agent


async def run_single_review(paper_content: str, paper_context: str = "") -> Review:
    """
    Runs a single review using the main reviewer agent, which internally uses tools.

    Args:
        paper_content: The content of the paper.
        paper_context: Optional context about the paper.

    Returns:
        The final Review object.
    """
    logger.info("Starting review process with ReviewerAgentWithTool...")

    # Construct the input for the agent (can be simple string or more structured if needed)
    # Adding context explicitly if the agent's prompt expects it separately,
    # otherwise, the agent might need instructions to use the context provided in the tool call.
    # For now, we assume the agent uses the context when calling the tool.
    agent_input = paper_content

    # Run the main reviewer agent
    result = await Runner.run(reviewer_agent, agent_input)

    final_review: Review = result.final_output
    logger.info("Review process completed.")
    logger.debug(
        f"Tool Usage Summary from Review: {final_review.summary_of_tool_usage}"
    )

    return final_review


async def main(paper_path: str, num_reviews: int):
    """
    Main method to run multiple reviews in parallel using the enhanced reviewer agent.

    Args:
        paper_path: Path to the paper file
        num_reviews: Number of reviews to get
    """
    dotenv.load_dotenv()

    # Read the paper
    logger.info(f"Reading paper from: {paper_path}")
    with open(paper_path, "r", encoding="utf-8") as f:
        paper_content = f.read()

    # --- Optional: Add logic here to fetch/define paper_context if available ---
    paper_context = ""  # Example: No context available
    # -------------------------------------------------------------------------

    # Create review jobs for parallel execution
    logger.info(f"Creating {num_reviews} review jobs...")
    review_jobs = []
    for i in range(num_reviews):
        logger.info(f"Adding review job {i + 1}/{num_reviews}")
        # Pass paper_content and paper_context to each review job
        review_jobs.append(run_single_review(paper_content, paper_context))

    # Run reviews in parallel
    logger.info(f"Running {num_reviews} reviews in parallel...")
    reviews: List[Review] = await asyncio.gather(*review_jobs)
    logger.info("All reviews completed.")

    # Save the results
    output_path = paper_path.replace(".md", "_reviews_tool_based.json")
    logger.info(f"Saving reviews to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert Review objects to dictionaries for JSON serialization
        reviews_dict = [review.model_dump() for review in reviews]
        json.dump(reviews_dict, f, indent=4)

    logger.info(f"Reviews successfully saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m openai_hackathon_wait.review_orchestrator <paper_path> <num_reviews>"
        )
        sys.exit(1)

    paper_path_arg = sys.argv[1]
    num_reviews_arg = int(sys.argv[2])

    # Configure logger (optional, but good practice)
    logger.add(sys.stderr, level="INFO")  # Change level to DEBUG for more details

    asyncio.run(main(paper_path_arg, num_reviews_arg))
