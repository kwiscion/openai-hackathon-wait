import asyncio
import json
import sys

import dotenv

# Import Agent/Runner from the SDK
from loguru import logger

from openai_hackathon_wait.review_orchestrator import review_orchestrator
from openai_hackathon_wait.utils.markdown_converter import convert

# Import agent creation functions and models directly

dotenv.load_dotenv()


async def main(paper_path: str):
    """
    Main method to run the review orchestrator.

    Args:
        paper_path: Path to the paper file
        num_reviews: Target number of reviews (currently unused as reviewer count is determined by selection)
    """
    if paper_path.endswith(".pdf"):
        convert(paper_path, force=False)
        paper_path = paper_path.replace(".pdf", "/manuscript.md")

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

    # Run the review orchestrator
    decision, reviews = await review_orchestrator(paper_content, paper_id)

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    try:
        with open(decision_output_path, "w", encoding="utf-8") as f:
            json.dump(decision.model_dump(), f, indent=4)
        logger.info(f"Decision saved to {decision_output_path}")
    except Exception as e:
        logger.error(f"Error saving decision to {decision_output_path}: {e}")

    # Save the reviews
    reviews_output_path = paper_path.replace(".md", "_reviews.json")
    try:
        with open(reviews_output_path, "w", encoding="utf-8") as f:
            json.dump([review.model_dump() for review in reviews], f, indent=4)
        logger.info(f"Reviews saved to {reviews_output_path}")
    except Exception as e:
        logger.error(f"Error saving reviews to {reviews_output_path}: {e}")

    return decision, reviews


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m openai_hackathon_wait.main <paper_path>")
        sys.exit(1)

    paper_path = sys.argv[1]
    asyncio.run(main(paper_path))
