import asyncio
import json
import sys

from agents import Runner, trace
from dotenv import load_dotenv
from openai import OpenAI

from openai_hackathon_wait.agents.reviewer import reviewer_agent

load_dotenv()

client = OpenAI()


async def main():
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    with trace("parallel_reviewer_agents"):
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(
                reviewer_agent,
                text,
            ),
            Runner.run(
                reviewer_agent,
                text,
            ),
            Runner.run(
                reviewer_agent,
                text,
            ),
        )

    output = [res_1.final_output, res_2.final_output, res_3.final_output]
    print(output)

    # Convert Review objects to dictionaries before serialization
    serializable_output = [review.model_dump() for review in output]

    output_file = path.replace(".md", "_reviews.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_output, f, indent=4)


asyncio.run(main())
