import asyncio
import json
import sys

import dotenv
from loguru import logger

from openai_hackathon_wait.agents.structure_validator import run_validator

from .publication_decision import PublicationDecisionOrchestrator
from .review_orchestrator import ReviewOrchestrator
from .review_synthesizer import ReviewSynthesizer

dotenv.load_dotenv()


async def main(paper_path: str, num_reviews: int):
    """
    Main method to run the review orchestrator.

    Args:
        paper_path: Path to the paper file
        num_reviews: Number of reviews to get
    """
    # Read the paper
    with open(paper_path, "r", encoding="utf-8") as f:
        paper = f.read()

    logger.info(f"Paper: {paper[:50]}...")

    # Run the structure validator
    structure_validator_result = await run_validator(
        paper_content=paper, auto_detect=True, grammar_check=True
    )

    # Save the structure validator result
    structure_validator_path = paper_path.replace(".md", "_structure_validator.json")
    with open(structure_validator_path, "w", encoding="utf-8") as f:
        json.dump(structure_validator_result, f, indent=4)

    logger.info(
        f"Structure validator result saved to {paper_path.replace('.md', '_structure_validator.json')}"
    )

    additional_analysis = [
        {"area": "structure and language", "review": structure_validator_result}
    ]

    # Run the review
    review_jobs = []
    for _ in range(num_reviews):
        orchestrator = ReviewOrchestrator()
        review_jobs.append(orchestrator.review_paper(paper, additional_analysis))

    reviews = await asyncio.gather(*review_jobs)

    # Save the results
    reviews_output_path = paper_path.replace(".md", "_reviews.json")
    with open(reviews_output_path, "w", encoding="utf-8") as f:
        reviews_dict = [review.model_dump() for review in reviews]
        json.dump(reviews_dict, f, indent=4)

    logger.info(f"Reviews saved to {reviews_output_path}")

    # Synthesize the reviews
    synthesizer = ReviewSynthesizer(reviews_output_path)
    synthesis = await synthesizer.synthesize()

    # Save the synthesis
    synthesis_output_path = reviews_output_path.replace(
        "_reviews.json", "_synthesis.json"
    )
    with open(synthesis_output_path, "w", encoding="utf-8") as f:
        json.dump(synthesis.model_dump(), f, indent=4)

    logger.info(f"Synthesis saved to {synthesis_output_path}")

    # Make the decision
    decision_orchestrator = PublicationDecisionOrchestrator(
        synthesis_output_path,
        paper_path,
        technical_analysis_path=structure_validator_path,
    )

    decision = await decision_orchestrator.make_decision()

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    with open(decision_output_path, "w", encoding="utf-8") as f:
        json.dump(decision.model_dump(), f, indent=4)

    logger.info(f"Decision saved to {decision_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m openai_hackathon_wait.review_orchestrator <paper_path> <num_reviews>"
        )
        sys.exit(1)

    paper_path = sys.argv[1]
    num_reviews = int(sys.argv[2])
    asyncio.run(main(paper_path, num_reviews))
