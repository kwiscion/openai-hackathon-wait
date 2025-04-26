from enum import Enum

from agents import Agent
from pydantic import BaseModel, Field

from .reviewer_assistant import reviewer_assistant_agent

PROMPT = (
    "You are a scientific reviewer. You are given a paper."
    "You need to review the paper and provide a review, including:"
    "- Strengths of the paper"
    "- Weaknesses of the paper"
    "- Comments on the paper"
    "- Overall rating (very good, good, fair, poor, very poor)"
    "- How confident you are in the rating (confident, unsure, not confident)"
    "- How much time you spent reviewing the paper"
    "- If there are any ethical concerns, please describe them."
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

    comments: str = Field(
        description="Detailed comments on the paper, integrating feedback on specific aspects."
    )

    rating: Rating = Field(description="The overall rating of the paper.")

    confidence: Confidence = Field(description="How confident you are in the rating.")

    ethical_concerns: str = Field(
        description="Any ethical concerns raised during the review."
    )


reviewer_assistant_tool = reviewer_assistant_agent.as_tool(
    tool_name="ReviewerAssistantTool",
    tool_description="A tool that can help the reviewer by providing a feedback on a specific aspect of the paper.",
)


def create_reviewer_agent(
    name: str = "ReviewerAgent",
    prompt: str = PROMPT,
    model: str = "gpt-4o-mini",
) -> Agent:
    return Agent(
        name=name,
        instructions=prompt,
        model=model,
        output_type=Review,
        tools=[reviewer_assistant_tool],  # Provide the tool function here
    )
