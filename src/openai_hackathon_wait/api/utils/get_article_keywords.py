from typing import List
from openai import AsyncOpenAI

from loguru import logger
from pydantic import BaseModel, Field


class ArticleKeywords(BaseModel):
    """List of keywords for academic search."""
    keywords: List[str] = Field(description="List of search keywords related to the input topics")


class ExpandedKeywords(BaseModel):
    """List of expanded keywords for academic search."""
    keywords: List[str] = Field(description="List of expanded search keywords related to the input topics")


async def get_article_keywords(article_text: str, number_of_keywords: int = 6) -> List[str]:
    """
    Generate keywords for arXiv search based on article text using OpenAI API.
    
    Args:
        article_text: The text of the article to search for 
    Returns:
        List of search keywords related to the input topics
    """
    client = AsyncOpenAI()
    
    system_message = """You are an expert researcher who helps expand search queries for scientific papers.
         Your task is to generate relevant keywords and phrases for academic databases searches.
         Search for keywords specified by authors of the article but also specify keywords related to the topics of the article.
         Consider synonyms, related concepts, more specific terms, broader terms, and alternative phrasings.
         Focus on terminology that would be used in academic papers and research articles."""
    
    human_message = f"Generate {number_of_keywords} search keywords related to these topics for an academic databases search: {article_text}"
    
    response = await client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message}
        ],
        text_format=ArticleKeywords,
    )
    
    keywords = response.output_parsed.keywords
    
    logger.info(f"Article keywords response: {keywords}")
    
    return list(set(keywords))


async def get_expanded_keywords(base_keywords: list[str], num_keywords: int = 5) -> List[str]:
    """
    Generate expanded keywords for arXiv search based on initial keywords using OpenAI API.
    
    Args:
        base_keywords: Initial list of search keywords
        num_keywords: Number of keywords to generate
        
    Returns:
        List of expanded search keywords
    """
    # Join keywords for better context
    keywords_str = ", ".join(base_keywords)
    
    client = AsyncOpenAI()
    
    system_message = """You are an expert researcher who helps expand search queries for scientific papers.
         Your task is to generate additional relevant keywords and phrases for academic databases searches.
         Consider synonyms, related concepts, more specific terms, broader terms, and alternative phrasings.
         Focus on terminology that would be used in academic papers and research articles."""
    
    human_message = f"Generate at most {num_keywords} additional search keywords related to these topics for an academic databases search: {keywords_str}"
    
    response = await client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message}
        ],
        text_format=ExpandedKeywords,
    )
    
    expanded_keywords = response.output_parsed.keywords
    
    logger.info(f"Expanded keywords response: {expanded_keywords}")
    
    return list(set(expanded_keywords + base_keywords))
