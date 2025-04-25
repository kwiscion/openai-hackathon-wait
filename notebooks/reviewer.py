import asyncio

from agents import Runner, trace
from dotenv import load_dotenv
from openai import OpenAI

from openai_hackathon_wait.reviewer import reviewer_agent

load_dotenv()

client = OpenAI()


async def main():
    path = "../data/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC.md"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    with trace("reviewer_agent"):
        result = await Runner.run(reviewer_agent, input=text)
    print(result.final_output)


asyncio.run(main())
