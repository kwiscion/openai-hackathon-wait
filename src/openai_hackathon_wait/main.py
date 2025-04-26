import asyncio
import json
import sys

import dotenv
from loguru import logger
from openai import AsyncOpenAI

from openai_hackathon_wait.agents.structure_validator import run_validator
from openai_hackathon_wait.create_context import create_context

# Import ReviewerFinder
from .agents.reviewer_finder import ReviewerFinder
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
    client = AsyncOpenAI()
    # Read the paper
    with open(paper_path, "r", encoding="utf-8") as f:
        paper = f.read()

    logger.info(f"Paper: {paper[:50]}...")
    paper_id = "paper_" + paper_path.split("/")[-1].split(".")[0][:50]

    # Create the context
    rag, paper_context = await create_context(client, paper_id, paper)

    # Save the context
    with open(paper_path.replace(".md", "_context.json"), "w", encoding="utf-8") as f:
        json.dump(paper_context, f, indent=4)

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

    # --- Find Reviewers --- 
    logger.info(f"Finding reviewers for paper: {paper_path}")
    finder = ReviewerFinder(paper_path)
    # Pass client if needed by ReviewerFinder internally, assuming it initializes its own for now
    selected_reviewers = await finder.find_reviewers()

    if not selected_reviewers:
        logger.error("Could not find reviewers. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(selected_reviewers)} reviewers: {list(selected_reviewers.keys())}")
    # Save the selected reviewers (optional, finder might already do it)
    finder.save_reviewers(selected_reviewers) 
    # --- End Find Reviewers ---

    # Run the review for each selected reviewer
    review_jobs = []
    # The num_reviews argument is now less relevant, we run one review per selected reviewer
    logger.info(f"Starting review process with {len(selected_reviewers)} selected reviewers...")
    for reviewer_name, system_prompt in selected_reviewers.items():
        logger.info(f"Initializing review orchestrator for: {reviewer_name}")
        orchestrator = ReviewOrchestrator(reviewer_name=reviewer_name, system_prompt=system_prompt)
        review_jobs.append(
            orchestrator.review_paper(paper, additional_analysis, paper_context)
        )

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
        literature_context_path=paper_context,
        technical_analysis_path=structure_validator_path,
    )

    decision = await decision_orchestrator.make_decision()

    # Save the decision
    decision_output_path = paper_path.replace(".md", "_decision.json")
    with open(decision_output_path, "w", encoding="utf-8") as f:
        json.dump(decision.model_dump(), f, indent=4)

    logger.info(f"Decision saved to {decision_output_path}")


if __name__ == "__main__":
    # Keep num_reviews for now, although it's not directly used to set the number of reviewers
    if len(sys.argv) < 3:
        print(
            "Usage: python -m openai_hackathon_wait.main <paper_path> <num_reviews (Note: actual number depends on reviewer selection)>"
        )
        sys.exit(1)

    paper_path = sys.argv[1]
    num_reviews = int(sys.argv[2]) # Not directly used, but kept for consistency 
    asyncio.run(main(paper_path, num_reviews))
