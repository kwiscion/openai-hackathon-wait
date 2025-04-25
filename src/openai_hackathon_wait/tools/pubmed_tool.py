import logging
from openai_hackathon_wait.api.pubmed import PubMedAgentTool
from agents import function_tool

logger = logging.getLogger(__name__)
pubmed_interface= PubMedAgentTool()


@function_tool
def pubmed_tool(query: str) -> str:
    """
    Search PubMed for biomedical literature and retrieve article summaries.
    
    Args:
        query: The search query for PubMed. Be specific to get relevant re
    prntaining the summaries of the top articles matching the query.
    """
    print(f"Searching PubMed for: {query}")
    try:
        return pubmed_interface.api_wrapper.run(query)
    except Exception as e:
        logger.error(f"Error searching PubMed: {str(e)}")
        return f"Error searching PubMed: {str(e)}"
