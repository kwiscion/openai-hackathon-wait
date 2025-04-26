from openai_hackathon_wait.create_context import context_agent
from agents import Runner, AsyncOpenAI
import asyncio
from openai_hackathon_wait.rag import rag
from openai_hackathon_wait.agents.reviewer_assistant import reviewer_assistant_agent
async def test_rag():

    client = AsyncOpenAI()
    with open("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00002v1/1601.00002v1.md", "r") as f:
        markdown_text1 = f.read()
    with open("/Users/karolinanowacka/hackathon/openai-hackathon-wait/data/1601.00003v1/1601.00003v1.md", "r") as f:
        markdown_text2 = f.read()
    
    result1 = await Runner.run(
        context_agent,
        input=markdown_text1
    )
    
    if hasattr(result1.final_output, 'articles_urls'):
        print(f"Found {len(result1.final_output.articles_urls)} relevant articles")
        for i, url in enumerate(result1.final_output.articles_urls, 1):
            print(f"{i}. {url}")

    await rag.create_vector_store()
    print(f"Created RAG with vector store name: {rag.vector_store_name}")
    
    # Add text to RAG
    print("\nAdding text to RAG...")
    await rag.add_text(markdown_text1[:5000])
    
    
    result2 = await Runner.run(
        reviewer_assistant_agent,
        input="Give examples of the papers?"
    )
    print(result2.final_output)
 
    

if __name__ == "__main__":
    result = asyncio.run(test_rag())
    