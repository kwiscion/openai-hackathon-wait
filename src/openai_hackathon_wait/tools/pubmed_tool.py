from openai_hackathon_wait.api.pubmed import PubMedAgentTool
from agents import function_tool
from loguru import logger

pubmed_interface= PubMedAgentTool()


@function_tool
def pubmed_tool(query: str) -> str:
    """
    Search PubMed for biomedical literature and retrieve article summaries.
    
    Args:
        query: The search query for PubMed. Be specific to get relevant re
    prntaining the summaries of the top articles matching the query.
    """
    try:
        return pubmed_interface.api_wrapper.run(query)
    except Exception as e:
        logger.error(f"Error searching PubMed: {str(e)}")
        return f"Error searching PubMed: {str(e)}"
