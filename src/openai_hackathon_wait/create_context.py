import os

import httpx
from agents import Agent, AsyncOpenAI, Runner
from anyio import TemporaryDirectory
from loguru import logger
from openai_hackathon_wait.api.deep_research import perform_deep_research

VECTOR_STORE_NAME = "global_paper_vector_store"
rag = None
from openai_hackathon_wait.tools.arxiv_tool import arxiv_search
from openai_hackathon_wait.tools.pubmed_tool import pubmed_tool


context_agent = Agent(
    name="Triage agent",
    instructions="""You are a helpful assistant that finds the most relevant papers from Arxiv or PubMed. 
    Decide which one is more relevant and return the results based on the proviede article text.""",
    tools=[arxiv_search, pubmed_tool ], # WE can define other handoffs like pubmed agent
    model="gpt-4o-mini"
)


async def summarize_deep_research(deep_research_text: str) -> str:
    client = AsyncOpenAI()
    response = await client.responses.create(
        instructions="""You are a helpful assistant that summarizes the provided text of the deep research.
        Return the summary focusing on the most important points relevant to the topic of the deep research output.
        The output will be than used to guide the review of the paper process.
        """,
        model="gpt-4o-mini",
        input=deep_research_text,
    )
    return response.output_text


async def create_context(
    client: AsyncOpenAI, article_text: str
) -> str:
    global rag
    
    # Initialize RAG if not already done
    if rag is None:
        rag = await create_rag()
        
    result = await Runner.run(
        context_agent,
        input=article_text,
    )
    article_urls = result.final_output

    deep_research_result = perform_deep_research(article_text)
    deep_research_text = deep_research_result.final_analysis
    summary = await summarize_deep_research(deep_research_text)


    article_urls = article_urls.articles_urls

    await rag.add_text(deep_research_text)

    # Use a temporary directory that will be automatically cleaned up
    async with TemporaryDirectory() as temp_dir:
        for article_url in article_urls:
            async with httpx.AsyncClient() as client:
                logger.info(f"Downloading {article_url}")
                response = await client.get(article_url)

                pdf_file_path = os.path.join(
                    temp_dir, f"{article_url.split('/')[-1]}.pdf"
                )
                with open(pdf_file_path, "wb", encoding="utf-8") as f:
                    f.write(response.content)

                await rag.upload_file(pdf_file_path)

    return summary
