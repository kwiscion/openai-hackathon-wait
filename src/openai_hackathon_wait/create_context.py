import os
import tempfile

import httpx
from agents import Agent, Runner, AsyncOpenAI
from loguru import logger
from pydantic import BaseModel

from openai_hackathon_wait.tools.arxiv_tool import arxiv_search
from openai_hackathon_wait.tools.pubmed_tool import pubmed_tool
from openai_hackathon_wait.rag import RAG


class ArxivSearchResult(BaseModel):
    articles_urls: list[str]


arxiv_agent = Agent(
    name="Arxiv agent",
    instructions="You provide information about the papers that are related to the query. Return between 15 and 25 results by default.",
    tools=[arxiv_search],
    model="gpt-4o-mini",
    output_type=ArxivSearchResult,
)

pubmed_agent = Agent(
    name="Pubmed agent",
    instructions="You provide information about the papers that are related to the query. Return between 15 and 25 results by default.",
    tools=[pubmed_tool],
    model="gpt-4o-mini",
    output_type=list[str],
)

triage_agent = Agent(
    name="Triage agent",
    instructions="""You are a helpful assistant that finds the most relevant papers from Arxiv or PubMed. 
    Decide which one is more relevant and return the results based on the proviede article text.""",
    handoffs=[arxiv_agent],
    model="gpt-4o-mini",
    output_type=ArxivSearchResult,
)


async def create_context(
    client: AsyncOpenAI, vector_store_name: str, article_text: str
) -> RAG:
    result = await Runner.run(
        triage_agent,
        input=article_text,
    )
    article_urls = result.final_output

    rag = RAG(vector_store_name)
    await rag.create_vector_store()

    article_urls = article_urls.articles_urls

    # Use a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        for article_url in article_urls:
            async with httpx.AsyncClient() as client:
                logger.info(f"Downloading {article_url}")
                response = await client.get(article_url)

                pdf_file_path = os.path.join(
                    temp_dir, f"{article_url.split('/')[-1]}.pdf"
                )
                with open(pdf_file_path, "wb") as f:
                    f.write(response.content)

                await rag.upload_file(pdf_file_path)

    return rag
