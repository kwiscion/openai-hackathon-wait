from enum import Enum

from agents import Agent, RunContextWrapper
from pydantic import BaseModel, Field

from .reviewer_assistant import reviewer_assistant_agent


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


class ReviewerContext(BaseModel):
    reviewer_persona: str = Field(description="The persona of the reviewer.")
    paper_content: str = Field(description="The content of the paper to review.")
    literature_context: str = Field(
        description="A summary of the literature context of the paper."
    )
    technical_context: str = Field(
        description="A summary of the technical context of the paper."
    )


reviewer_assistant_tool = reviewer_assistant_agent.as_tool(
    tool_name="ReviewerAssistantTool",
    tool_description="A tool that can help the reviewer by providing a feedback on a specific aspect of the paper.",
)


def dynamic_instructions(
    wrapper: RunContextWrapper[ReviewerContext],
    agent: Agent[ReviewerContext],
) -> str:
    ctx = wrapper.context
    return f"""
    You are a scientific reviewer. Your persona is:
    {ctx.reviewer_persona}

    Your task is to review the paper and provide a review.

    You are given:
    - A paper content: {ctx.paper_content}
    - A literature context: {ctx.literature_context}
    - A technical context: {ctx.technical_context}
    """


def create_reviewer_agent(
    name: str = "ReviewerAgent",
    model: str = "gpt-4o-mini",
) -> Agent[ReviewerContext]:
    return Agent[ReviewerContext](
        name=name,
        instructions=dynamic_instructions,
        model=model,
        output_type=Review,
        tools=[reviewer_assistant_tool],  # Provide the tool function here
    )
