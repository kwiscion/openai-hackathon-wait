from typing import List

from agents import Agent
from pydantic import BaseModel, Field

PROMPT = (
    "You are a scientific reviewer. You are given a paper."
    "Based on your expertise and the paper, please suggest the areas to review the paper."
    "When deciding the areas to review, please take into account paper's domain and type."
    "You should suggest 3-5 areas to review."
)


class ReviewAreas(BaseModel):
    areas: List[str] = Field(description="The areas to review the paper.")


review_planner_agent = Agent(
    name="ReviewerAgent",
    instructions=PROMPT,
    model="gpt-4o-mini",
    output_type=ReviewAreas,
)
