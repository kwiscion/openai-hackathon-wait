from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict
from agents import function_tool

from openai_hackathon_wait.api.arxiv_search_api import ArxivSearchWrapper

arxiv_search_wrapper = ArxivSearchWrapper()


class ArxivSearchConfig(TypedDict):
    query: str
    max_results: int = 5
    category: Optional[str] = None
    sort_by: str = "relevance"


@function_tool
def arxiv_search(
    config: ArxivSearchConfig
) -> Dict[str, Union[List[Dict[str, Any]], str]]:
    """
    Searches the arXiv database for scientific papers based on keywords.
    
    This tool queries arXiv.org to find academic papers matching the search terms
    and returns metadata including URLs to promising papers.
    
    Args:
        query: Keywords to search for in arXiv papers
        max_results: Number of results to return (default: 5, max: 20)
        category: Optional arXiv category to limit search (e.g., 'cs.AI', 'physics')
        sort_by: How to sort results ('relevance', 'lastUpdatedDate', 'submittedDate')
    
    Returns:
        Dictionary containing:
        - 'papers': List of paper metadata dictionaries with URLs
        - 'error': Error message if search failed (only present if there was an error)
    """
    # Validate inputs
    max_results = min(max(1, max_results), 20)  # Clamp between 1 and 20
    sort_by = sort_by if sort_by in ["relevance", "lastUpdatedDate", "submittedDate"] else "relevance"
    
    # Set the number of results to return
    arxiv_search_wrapper.top_k_results = max_results
    arxiv_search_wrapper.sort_by = sort_by
    
    # Perform the search
    papers = arxiv_search_wrapper.search(query=config["query"], category=config["category"])
    
    if not papers:
        return {
            "papers": [],
            "error": f"No relevant papers found on arXiv for query: '{config['query']}'",
        }
    
    # Format the response
    return {
        "papers": papers
    } 
