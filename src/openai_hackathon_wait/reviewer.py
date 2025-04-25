from agents import Agent
from pydantic import BaseModel

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

# class Rating(str, Enum):
#     VERY_GOOD = "very good"
#     GOOD = "good"
#     FAIR = "fair"
#     POOR = "poor"
#     VERY_POOR = "very poor"


class Review(BaseModel):
    strengths: str
    "The strengths of the paper."

    weaknesses: str
    "The weaknesses of the paper."

    comments: str
    "Comments on the paper."

    rating: str
    "The rating of the paper."

    confidence: str
    "How confident you are in the rating."

    time_spent: str
    "How much time you spent reviewing the paper."

    ethical_concerns_flag: bool
    "If there are any ethical concerns."


reviewer_agent = Agent(
    name="ReviewerAgent",
    instructions=PROMPT,
    model="gpt-4o-mini",
    output_type=Review,
)
