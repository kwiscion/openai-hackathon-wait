from typing import Any, Dict, List, Optional, Union, TypedDict
import os
import asyncio
from agents import function_tool, RunContextWrapper
from loguru import logger

from openai_hackathon_wait.api.arxiv_articles import ArxivAPI
from openai_hackathon_wait.api.utils.get_article_keywords import get_expanded_keywords, get_article_keywords


class ArxivSearchArgs(TypedDict):
    """Arguments for the arXiv search function."""
    article_text: str
    max_results: int
    sort_by: str


def default_tool_error_function(error: Exception) -> str:
    """Default error handler for arXiv search tool."""
    error_message = str(error)
    logger.error(f"Error in arXiv search tool: {error_message}")
    return f"Error searching arXiv: {error_message}. Please try with different parameters or a more specific query."


@function_tool(failure_error_function=default_tool_error_function)
async def arxiv_search(
    ctx: RunContextWrapper[Any],
    article_text: str,
    max_results: Optional[int] = None,
    sort_by: Optional[str] = None,
) -> list[str]:
    """
    Searches the arXiv database for scientific papers based on keywords extracted from the article text.
    
    This tool analyzes the article text to extract relevant search terms, expands them with related concepts,
    and queries arXiv.org to find academic papers matching these terms. Results include metadata such as
    title, authors, abstract, and URLs to access the papers.
    
    Args:
        article_text: The text or query to find relevant scientific papers for
        max_results: Number of results to return (default: 25, max:30)
        sort_by: How to sort results - options: 'relevance', 'lastUpdatedDate', 'submittedDate'
    
    Returns:
        Dictionary containing:
        - 'papers': List of paper metadata dictionaries with URLs to abstracts and PDFs
        - 'error': Error message if search failed (only present if there was an error)
    """
    # Input validation
    if not article_text or not article_text.strip():
        return {
            "papers": [],
            "error": "Empty article text provided. Please provide some content to search for."
        }
    
    max_results = 25 if max_results is None else max_results
    max_results = 1
    sort_by = "relevance" if sort_by is None else sort_by
    
    max_results = min(max(1, max_results), 30)

    logger.info(f"Max results: {max_results}")
    logger.info(f"Sort by: {sort_by}")

    valid_sort_options = ["relevance", "lastUpdatedDate", "submittedDate"]
    sort_by = sort_by if sort_by in valid_sort_options else "relevance"

    # Create API client
    arxiv_api = ArxivAPI(
        max_results=max_results,
        sort_by=sort_by,
    )

    try:
        logger.info(f"Starting arXiv search for: {article_text[:100]}{'...' if len(article_text) > 100 else ''}")
        
        article_keywords = await get_article_keywords(article_text)
        logger.info(f"Extracted keywords: {article_keywords}")
        
        expanded_keywords = await get_expanded_keywords(article_keywords)
        logger.info(f"Expanded keywords: {expanded_keywords}")
        
        papers_urls = await arxiv_api.search_articles(expanded_keywords)
        
        logger.info(f"Found {len(papers_urls)} papers urls on arXiv matching the query")

        logger.info(f"Papers: {papers_urls}")
        
        if not papers_urls:
            return []
        
        # fetch the pdfs and get the content
        papers_content = await asyncio.gather(*[arxiv_api.fetch_pdf_and_get_article_content(paper_url) for paper_url in papers_urls])
        
        return papers_content
        
    except Exception as e:
        logger.error(f"Error in arXiv search: {str(e)}")
        return []
