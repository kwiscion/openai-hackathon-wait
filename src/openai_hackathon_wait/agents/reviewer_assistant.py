from agents import Agent

PROMPT = (
    "You are a scientific reviewer. You are given a paper."
    "Your task is to help the reviewer by providing a feedback on a specific aspect of the paper."
)


reviewer_assistant_agent = Agent(
    name="ReviewerAssistantAgent",
    instructions=PROMPT,
    model="gpt-4o-mini",
)
