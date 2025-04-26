import asyncio
from typing import Any, Dict, List

from agents import Agent, Runner
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
    def __init__(self, reviewer_name: str = "GeneralReviewer", system_prompt: str = ""):
        # If a custom system prompt is provided, create a new Agent instance
        if system_prompt:
            self.reviewer_name = reviewer_name
            self.reviewer_agent = Agent(
                name=f"{reviewer_name}ReviewerAgent", # Use the name
                instructions=system_prompt, # Use the custom prompt
                output_type=Review, # Keep the original output type
                model=reviewer_agent.model, # Keep the original model
            )
            logger.info(f"Initialized orchestrator for reviewer: {reviewer_name}")
        else:
            # Otherwise, use the default reviewer agent
            self.reviewer_name = "GeneralReviewer"
            self.reviewer_agent = reviewer_agent
            logger.info("Initialized orchestrator with default reviewer.")
            
        # Assistant agent remains the same
        self.reviewer_assistant_agent = reviewer_assistant_agent

    async def get_assistant_reviews(
        self, paper: str, analysis_areas: List[str], paper_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get reviews from multiple assistant agents in parallel.
        """
        logger.info("Starting assistant reviews...")

        # Create tasks for parallel execution
        tasks = []
        for analysis_area in analysis_areas:
            logger.info(f"Creating task for reviewing {analysis_area}...")
            prompt = f"""
            Please review the following paper and provide a feedback on the {analysis_area}.
            
            Paper:
            
            {paper}
            
            Paper context:
            
            {paper_context}
            """
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
        self,
        paper: str,
        assistant_reviews: List[Dict[str, Any]],
        paper_context: str = "",
    ) -> Review:
        """
        Get the main review from the reviewer agent, incorporating feedback from assistant reviews.
        """
        logger.info(f"Getting main review from reviewer agent: {self.reviewer_name}...")

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
        
        Paper context:
        
        {paper_context}
        """

        # Get the main review
        main_review = await Runner.run(self.reviewer_agent, prompt)
        return main_review.final_output

    async def get_analysis_areas(
        self, paper: str, paper_context: str = ""
    ) -> List[str]:
        """
        Get the analysis areas for the paper.
        """
        prompt = f"""
        Please review the following paper and provide a list of areas to review.
        Exclude areas that will be included by default:
        {", ".join(ANALYSIS_AREAS)}
        
        Paper:
        
        {paper}
        
        Paper context:
        
        {paper_context}
        """
        result = await Runner.run(review_planner_agent, prompt)
        areas = result.final_output.areas
        return areas + ANALYSIS_AREAS

    async def review_paper(
        self,
        paper: str,
        additional_analysis: List[Dict[str, Any]] = [],
        paper_context: str = "",
    ) -> FullReview:
        """
        Review a paper using multiple agents in parallel and then get a main review.

        Args:
            paper: The paper to review
            additional_analysis: Additional analysis done by other agents
            paper_context: Context about the paper
        """
        # Get the analysis areas
        analysis_areas = await self.get_analysis_areas(paper, paper_context)
        logger.info(f"Analysis areas: {analysis_areas}")

        # Get reviews from assistant agents
        assistant_reviews = await self.get_assistant_reviews(
            paper, analysis_areas, paper_context
        )

        # Add additional analysis to the assistant reviews
        for analysis in additional_analysis:
            assistant_reviews.append(
                {"area": analysis["area"], "review": analysis["review"]}
            )

        # Get the main review
        main_review = await self.get_main_review(
            paper, assistant_reviews, paper_context
        )

        # Return both assistant reviews and main review
        return FullReview(
            assistant_feedback=assistant_reviews,
            review=main_review,
            # Add reviewer name to the output if needed, for now just logging
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
