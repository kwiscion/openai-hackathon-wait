from typing import Optional, List
from agents import function_tool, RunContextWrapper
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import os
from anyio import TemporaryDirectory

from openai_hackathon_wait.rag import RAG

# Maintain a dictionary of RAG instances to reuse them by name
rag_instances = {}

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

@function_tool
async def create_rag_store(
    ctx: RunContextWrapper,
    vector_store_name: str,
    model: Optional[str] = None
) -> str:
    """
    Creates a new RAG (Retrieval Augmented Generation) vector store.
    
    Args:
        vector_store_name: Name to identify this vector store
        model: Optional model to use for completions (defaults to gpt-4o-mini)
        
    Returns:
        A confirmation message that the vector store was created
    """
    try:
        logger.info(f"Creating RAG vector store: {vector_store_name}")
        
        # Use existing instance if available
        if vector_store_name in rag_instances:
            return f"RAG vector store '{vector_store_name}' already exists"
        
        # Create new RAG instance
        rag = RAG(vector_store_name, model=model or "gpt-4o-mini")
        await rag.create_vector_store()
        rag_instances[vector_store_name] = rag
        
        return f"RAG vector store '{vector_store_name}' created successfully"
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error creating RAG vector store: {error_message}")
        return f"Error creating RAG vector store: {error_message}"


@function_tool
async def create_rag_from_arxiv(
    ctx: RunContextWrapper,
    vector_store_name: str,
    article_urls: List[str],
    text_content: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Creates a new RAG vector store and populates it with content from arxiv articles and optional text.
    
    Args:
        vector_store_name: Name to identify this vector store
        article_urls: List of URLs to arxiv articles (from arxiv_agent results)
        text_content: Optional additional text content to add to the vector store
        model: Optional model to use for completions (defaults to gpt-4o-mini)
        
    Returns:
        A confirmation message about the creation and population of the vector store
    """
    try:
        logger.info(f"Creating RAG vector store from arxiv articles: {vector_store_name}")
        
        # Use existing instance if available
        if vector_store_name in rag_instances:
            return f"RAG vector store '{vector_store_name}' already exists"
        
        # Create new RAG instance
        rag = RAG(vector_store_name, model=model or "gpt-4o-mini")
        await rag.create_vector_store()
        rag_instances[vector_store_name] = rag
        
        # Add optional text content if provided
        if text_content:
            logger.info(f"Adding text content to vector store: {vector_store_name}")
            await rag.add_text(text_content)
        
        # Add articles from URLs
        logger.info(f"Adding {len(article_urls)} articles to vector store: {vector_store_name}")
        
        article_count = 0
        error_count = 0
        
        # Use a temporary directory that will be automatically cleaned up
        async with TemporaryDirectory() as temp_dir:
            for i, article_url in enumerate(article_urls):
                try:
                    logger.info(f"Downloading article {i+1}/{len(article_urls)}: {article_url}")
                    async with httpx.AsyncClient() as client:
                        response = await client.get(article_url)
                        
                        if response.status_code != 200:
                            logger.error(f"Failed to download article: {article_url}, status code: {response.status_code}")
                            error_count += 1
                            continue
                        
                        pdf_file_path = os.path.join(
                            temp_dir, f"article_{i}_{article_url.split('/')[-1]}.pdf"
                        )
                        with open(pdf_file_path, "wb") as f:
                            f.write(response.content)
                        
                        await rag.upload_file(pdf_file_path)
                        article_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing article {article_url}: {str(e)}")
                    error_count += 1
        
        success_message = f"RAG vector store '{vector_store_name}' created successfully with {article_count} articles"
        if error_count > 0:
            success_message += f" ({error_count} articles failed to process)"
        if text_content:
            success_message += " and additional text content"
            
        return success_message
    
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error creating RAG vector store from arxiv articles: {error_message}")
        return f"Error creating RAG vector store from arxiv articles: {error_message}"


@function_tool
async def upload_file_to_rag(
    ctx: RunContextWrapper,
    vector_store_name: str,
    file_path: str
) -> str:
    """
    Uploads a file to an existing RAG vector store.
    
    Args:
        vector_store_name: Name of the vector store to upload to
        file_path: Path to the file to upload
        
    Returns:
        A confirmation message that the file was uploaded
    """
    try:
        if vector_store_name not in rag_instances:
            return f"RAG vector store '{vector_store_name}' not found. Create it first with create_rag_store."
        
        rag = rag_instances[vector_store_name]
        logger.info(f"Uploading file {file_path} to RAG vector store: {vector_store_name}")
        
        await rag.upload_file(file_path)
        return f"File {file_path} uploaded successfully to RAG vector store '{vector_store_name}'"
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error uploading file to RAG vector store: {error_message}")
        return f"Error uploading file to RAG vector store: {error_message}"


@function_tool
async def add_text_to_rag(
    ctx: RunContextWrapper,
    vector_store_name: str,
    text: str
) -> str:
    """
    Adds text content to an existing RAG vector store.
    
    Args:
        vector_store_name: Name of the vector store to add text to
        text: Text content to add to the vector store
        
    Returns:
        A confirmation message that the text was added
    """
    try:
        if vector_store_name not in rag_instances:
            return f"RAG vector store '{vector_store_name}' not found. Create it first with create_rag_store."
        
        rag = rag_instances[vector_store_name]
        logger.info(f"Adding text to RAG vector store: {vector_store_name}")
        
        await rag.add_text(text)
        return f"Text content added successfully to RAG vector store '{vector_store_name}'"
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error adding text to RAG vector store: {error_message}")
        return f"Error adding text to RAG vector store: {error_message}"


@function_tool
async def query_rag_store(
    ctx: RunContextWrapper,
    vector_store_name: str,
    query: str
) -> list[str]:
    """
    Queries an existing RAG vector store with a natural language question.
    
    Args:
        vector_store_name: Name of the vector store to query
        query: Natural language question to ask the RAG system
        
    Returns:
        List of text responses from the RAG system
    """
    try:
        if vector_store_name not in rag_instances:
            return [f"RAG vector store '{vector_store_name}' not found. Create it first with create_rag_store."]
        
        rag = rag_instances[vector_store_name]
        logger.info(f"Querying RAG vector store '{vector_store_name}' with: {query}")
        
        results = await rag.ask_question(query)
        logger.info(f"Got {len(results)} results from RAG query")
        
        return results
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error querying RAG vector store: {error_message}")
        return [f"Error querying RAG vector store: {error_message}"]


@function_tool
async def delete_rag_store(
    ctx: RunContextWrapper,
    vector_store_name: str
) -> str:
    """
    Deletes an existing RAG vector store.
    
    Args:
        vector_store_name: Name of the vector store to delete
        
    Returns:
        A confirmation message that the vector store was deleted
    """
    try:
        if vector_store_name not in rag_instances:
            return f"RAG vector store '{vector_store_name}' not found"
        
        rag = rag_instances[vector_store_name]
        logger.info(f"Deleting RAG vector store: {vector_store_name}")
        
        await rag.delete_vector_store()
        del rag_instances[vector_store_name]
        
        return f"RAG vector store '{vector_store_name}' deleted successfully"
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error deleting RAG vector store: {error_message}")
        return f"Error deleting RAG vector store: {error_message}" 