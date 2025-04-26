import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from agents import Agent, Runner
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from openai_hackathon_wait.agents.reviewer import Review

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Ensure OpenAI client is initialized
client = OpenAI()


# Define rating and confidence enums
class Rating(str, Enum):
    VERY_GOOD = "very good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very poor"


class Confidence(str, Enum):
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    NOT_CONFIDENT = "not confident"


# Define Pydantic model for synthesized review
class SynthesizedReview(BaseModel):
    synthesized_strengths: str = Field(
        description="Synthesis of all reviewer-identified strengths."
    )
    synthesized_weaknesses: str = Field(
        description="Synthesis of all reviewer-identified weaknesses."
    )
    overall_assessment: str = Field(
        description="A comprehensive assessment synthesizing reviewer comments."
    )
    editorial_recommendation: str = Field(
        description="Editorial recommendation based on reviews (e.g., Accept, Minor Revision, Major Revision, Reject)."
    )
    confidence_assessment: str = Field(
        description="Assessment of overall confidence in reviews."
    )
    ethical_concerns: str = Field(
        description="Summary of any ethical concerns raised by reviewers."
    )
    has_ethical_concerns: bool = Field(
        description="Whether any ethical concerns were raised."
    )


# Create the synthesizer agent
PROMPT = """
You are an expert academic editor synthesizing peer reviews of a scientific manuscript.
Your task is to analyze multiple reviews, identify patterns of agreement and disagreement,
and provide a comprehensive synthesis that will guide the editorial decision.

When synthesizing reviews:
1. Identify common strengths noted across multiple reviews
2. Identify common weaknesses or concerns
3. Note areas where reviewers disagree
4. Consider the confidence levels of each reviewer
5. Weigh ethical concerns appropriately
6. Formulate a balanced assessment that represents the consensus view
7. Provide a clear editorial recommendation based on review content

Your synthesis should be fair, balanced, and accurately represent the full range of reviewer perspectives.
Avoid overemphasizing either positive or negative feedback unless there is clear consensus.
    """


def create_synthesizer_agent(model: str = "gpt-4o-mini"):
    return Agent(
        name="ReviewSynthesisAgent",
        instructions=PROMPT,
        output_type=SynthesizedReview,
        model=model,
    )


async def run_synthesizer_agent(
    reviews: List[Review], model: str = "gpt-4o-mini"
) -> Optional[SynthesizedReview]:
    try:
        synthesizer = create_synthesizer_agent(model=model)
        formatted_reviews = "Reviews: " + json.dumps(
            [review.model_dump() for review in reviews]
        )
        result = await Runner.run(synthesizer, formatted_reviews)
        return result.final_output_as(SynthesizedReview)
    except Exception as e:
        logger.error(f"Error during review synthesis: {e}")
        return None


class ReviewSynthesizer:
    def __init__(self, reviews_file_path: str):
        """Initialize the review synthesizer with path to the reviews JSON file."""
        self.reviews_file_path = reviews_file_path

    async def synthesize(self, reviews: List[Review]) -> Optional[SynthesizedReview]:
        """Synthesize the reviews using the agent."""
        if not reviews:
            logger.warning("No reviews found or error loading reviews.")
            return None

        # Format the reviews as input for the agent
        formatted_reviews = json.dumps([review.model_dump() for review in reviews])

        # Create the synthesis agent
        synthesis_agent = create_synthesizer_agent()

        # Run the synthesis agent
        result = await Runner.run(synthesis_agent, formatted_reviews)

        # Return the synthesized review
        return result.final_output_as(SynthesizedReview)

    def save_synthesis(
        self, synthesis: SynthesizedReview, output_path: Optional[str] = None
    ):
        """Save the synthesized review to a JSON file."""
        if output_path is None:
            # Create output filename based on input filename
            input_path = Path(self.reviews_file_path)
            output_path = str(input_path.parent / f"{input_path.stem}_synthesis.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(synthesis.model_dump(), f, indent=4)

        logger.info(f"Synthesis saved to {output_path}")


async def main():
    # Path to the reviews JSON file - use path relative to script location
    script_dir = Path(__file__).parent.parent.parent  # Go up to the project root
    reviews_file = (
        script_dir
        / "data/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC_reviews.json"
    )

    # Initialize and run the synthesizer
    synthesizer = ReviewSynthesizer(str(reviews_file))
    synthesis = await synthesizer.synthesize()

    if synthesis:
        # Save the synthesis
        synthesizer.save_synthesis(synthesis)

        # Log the synthesis
        logger.info("\n=== SYNTHESIZED REVIEW ===")
        logger.info(f"STRENGTHS:\n{synthesis.synthesized_strengths}\n")
        logger.info(f"WEAKNESSES:\n{synthesis.synthesized_weaknesses}\n")
        logger.info(f"ASSESSMENT:\n{synthesis.overall_assessment}\n")
        logger.info(f"RECOMMENDATION:\n{synthesis.editorial_recommendation}\n")
        logger.info(f"CONFIDENCE:\n{synthesis.confidence_assessment}\n")
        logger.info(f"ETHICAL CONCERNS:\n{synthesis.ethical_concerns}\n")


if __name__ == "__main__":
    asyncio.run(main())
