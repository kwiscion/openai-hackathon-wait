from typing import List

from loguru import logger
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage



class ArticleKeywords(BaseModel):
    """List of keywords for academic search."""
    keywords: List[str] = Field(description="List of search keywords related to the input topics")


class ExpandedKeywords(BaseModel):
    """List of expanded keywords for academic search."""
    keywords: List[str] = Field(description="List of expanded search keywords related to the input topics")


async def get_article_keywords(article_text: str) -> List[str]:
    """
    Generate keywords for arXiv search based on article text using LangChain.
    
    Args:
        article_text: The text of the article to search for 
    Returns:
        List of search keywords related to the input topics
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)
    structured_llm = model.with_structured_output(ArticleKeywords)

    messages = [
        SystemMessage(content="""You are an expert researcher who helps expand search queries for scientific papers.
             Your task is to generate relevant keywords and phrases for academic databases searches.
             Search for keywords specified by authors of the article but also specify keywords related to the topics of the article.
             Consider synonyms, related concepts, more specific terms, broader terms, and alternative phrasings.
             Focus on terminology that would be used in academic papers and research articles."""),
        HumanMessage(content=f"Generate search keywords related to these topics for an academic databases search: {article_text}")
    ]

    response = await structured_llm.ainvoke(messages)
    logger.info(f"Article keywords response: {response}")
    
    return list(set(response.keywords))


async def get_expanded_keywords(base_keywords: list[str], num_keywords: int = 10) -> List[str]:
    """
    Generate expanded keywords for arXiv search based on initial keywords using LangChain.
    
    Args:
        base_keywords: Initial list of search keywords
        num_keywords: Number of keywords to generate
        
    Returns:
        List of expanded search keywords
    """
    # Join keywords for better context
    keywords_str = ", ".join(base_keywords)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)
    structured_llm = model.with_structured_output(ExpandedKeywords)

    messages = [
        SystemMessage(content="""You are an expert researcher who helps expand search queries for scientific papers.
             Your task is to generate additional relevant keywords and phrases for academic databases searches.
             Consider synonyms, related concepts, more specific terms, broader terms, and alternative phrasings.
             Focus on terminology that would be used in academic papers and research articles."""),
        HumanMessage(content=f"Generate {num_keywords} additional search keywords related to these topics for an academic databases search: {keywords_str}")
    ]

    response = await structured_llm.ainvoke(messages)
    logger.info(f"Expanded keywords response: {response}")
    
    return list(set(response.keywords + base_keywords))
