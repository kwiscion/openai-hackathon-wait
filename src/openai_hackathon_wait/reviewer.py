from enum import Enum

from agents import Agent
from pydantic import BaseModel, Field

PROMPT = (
    "You are a scientific reviewer. You are given a paper."
    "You need to review the paper and provide a review, including:"
    "- Strengths of the paper"
    "- Weaknesses of the paper"
    "- Comments on the paper"
    "- Overall rating (very good, good, fair, poor, very poor)"
    "- How confident you are in the rating (confident, unsure, not confident)"
    "- How much time you spent reviewing the paper"
    "- If there are any ethical concerns"
)


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


class Review(BaseModel):
    strengths: str = Field(description="The strengths of the paper.")

    weaknesses: str = Field(description="The weaknesses of the paper.")

    comments: str = Field(description="Comments on the paper.")

    rating: Rating = Field(description="The rating of the paper.")

    confidence: Confidence = Field(description="How confident you are in the rating.")

    ethical_concerns_flag: bool = Field(
        description="If there are any ethical concerns."
    )


reviewer_agent = Agent(
    name="ReviewerAgent",
    instructions=PROMPT,
    model="gpt-4o-mini",
    output_type=Review,
)
