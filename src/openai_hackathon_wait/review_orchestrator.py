import asyncio
from typing import Any, Dict, List

from agents import Runner
from loguru import logger
from pydantic import BaseModel

from .agents.review_planner import review_planner_agent
from .agents.reviewer import Review, reviewer_agent
from .agents.reviewer_assistant import reviewer_assistant_agent


class FullReview(BaseModel):
    assistant_feedback: List[Dict[str, Any]]
    review: Review


ANALYSIS_AREAS = [
    "Novelty",
    "Ethical Concerns",
    "Methodology",
]


class ReviewOrchestrator:
    def __init__(self):
        self.reviewer_agent = reviewer_agent
        self.reviewer_assistant_agent = reviewer_assistant_agent

    async def get_assistant_reviews(
        self, paper: str, analysis_areas: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get reviews from multiple assistant agents in parallel.
        """
        logger.info("Starting assistant reviews...")

        # Create tasks for parallel execution
        tasks = []
        for analysis_area in analysis_areas:
            logger.info(f"Creating task for reviewing {analysis_area}...")
            prompt = f"Please review the following paper and provide a feedback on the {analysis_area}.\n\nPaper:\n\n{paper}"
            tasks.append(Runner.run(self.reviewer_assistant_agent, prompt))

        # Run all tasks in parallel
        logger.info("Running all assistant review tasks in parallel...")
        reviews = await asyncio.gather(*tasks)

        # Process the results
        results = []
        for analysis_area, review in zip(analysis_areas, reviews):
            logger.info(f"Processing review for {analysis_area}...")
            results.append({"area": analysis_area, "review": review.final_output})

        return results

    async def get_main_review(
        self, paper: str, assistant_reviews: List[Dict[str, Any]]
    ) -> Review:
        """
        Get the main review from the reviewer agent, incorporating feedback from assistant reviews.
        """
        logger.info("Getting main review from reviewer agent...")

        # Format assistant reviews for the prompt
        assistant_feedback = "\n\n".join(
            [
                f"Feedback on {review['area']}:\n{review['review']}"
                for review in assistant_reviews
            ]
        )

        # Create the prompt for the main reviewer
        prompt = f"""
        Please review the following paper and provide a comprehensive review.
        
        Here is feedback from multiple assistant reviewers:
        
        {assistant_feedback}
        
        Paper:
        
        {paper}
        """

        # Get the main review
        main_review = await Runner.run(self.reviewer_agent, prompt)
        return main_review.final_output

    async def get_analysis_areas(self, paper: str) -> List[str]:
        """
        Get the analysis areas for the paper.
        """
        prompt = f"""
        Please review the following paper and provide a list of areas to review.
        Exclude areas that will be included by default:
        {", ".join(ANALYSIS_AREAS)}
        
        Paper:
        
        {paper}
        """
        result = await Runner.run(review_planner_agent, prompt)
        areas = result.final_output.areas
        return areas + ANALYSIS_AREAS

    async def review_paper(self, paper: str) -> FullReview:
        """
        Review a paper using multiple agents in parallel and then get a main review.
        """
        # Get the analysis areas
        analysis_areas = await self.get_analysis_areas(paper)
        logger.info(f"Analysis areas: {analysis_areas}")

        # Get reviews from assistant agents
        assistant_reviews = await self.get_assistant_reviews(paper, analysis_areas)

        # Get the main review
        main_review = await self.get_main_review(paper, assistant_reviews)

        # Return both assistant reviews and main review
        return FullReview(
            assistant_feedback=assistant_reviews,
            review=main_review,
        )


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

    # Run the review
    review_jobs = []
    for _ in range(num_reviews):
        orchestrator = ReviewOrchestrator()
        review_jobs.append(orchestrator.review_paper(paper))

    reviews = await asyncio.gather(*review_jobs)

    # Save the results
    output_path = paper_path.replace(".md", "_reviews.json")
    with open(output_path, "w", encoding="utf-8") as f:
        import json

        reviews_dict = [review.model_dump() for review in reviews]

        json.dump(reviews_dict, f, indent=4)

    logger.info(f"Reviews saved to {output_path}")
    return reviews


if __name__ == "__main__":
    import sys

    import dotenv

    dotenv.load_dotenv()

    if len(sys.argv) < 3:
        print(
            "Usage: python -m openai_hackathon_wait.review_orchestrator <paper_path> <num_reviews>"
        )
        sys.exit(1)

    paper_path = sys.argv[1]
    num_reviews = int(sys.argv[2])
    asyncio.run(main(paper_path, num_reviews))
