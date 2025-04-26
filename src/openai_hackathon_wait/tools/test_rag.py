from openai_hackathon_wait.agents.image_analysis.markdown_converter import convert
from openai_hackathon_wait.create_context import context_agent

markdown_text1 = convert("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00002v1.pdf")
markdown_text2 = convert("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00003v1.pdf")

with open("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00002v1/1601.00002v1.md", "r") as f:
    markdown_text1 = f.read()

with open("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00003v1/1601.00003v1.md", "r") as f:
    markdown_text2 = f.read()

context_agent.run(markdown_text1)



if __name__ == "__main__":
    print(markdown_text1)
    print(markdown_text2)