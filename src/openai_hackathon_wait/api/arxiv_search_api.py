"""
ArXiv Search Tool for OpenAI Agents.

This tool searches for academic papers on arXiv based on keywords and returns
URLs of promising papers, along with their metadata.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
logger = logging.getLogger(__name__)


class ArxivSearchWrapper(BaseModel):
    """
    Wrapper for ArXiv API search capabilities.
    
    Uses the arxiv Python package to search for papers and return relevant metadata
    including URLs to the papers.
    """
    
    top_k_results: int = Field(
        default=5, 
        description="Number of top results to return"
    )
    max_query_length: int = Field(
        default=300, 
        description="Maximum query length"
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')"
    )
    sort_order: str = Field(
        default="descending",
        description="Sort direction ('ascending' or 'descending')"
    )
    
    # Fields that will be populated during initialization
    arxiv_search: Any = None
    arxiv_exceptions: Any = None
    arxiv_sort_criterion: Any = None
    arxiv_sort_order: Any = None
    
    @model_validator(mode="after")
    def validate_environment(self) -> "ArxivSearchWrapper":
        """Validate that the required packages are installed."""
        try:
            import arxiv
            
            # Set up the search object and exceptions
            self.arxiv_search = arxiv.Search
            self.arxiv_exceptions = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            )
            
            # Set up the sort criteria
            self.arxiv_sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
            }.get(self.sort_by, arxiv.SortCriterion.Relevance)
            
            self.arxiv_sort_order = {
                "ascending": arxiv.SortOrder.Ascending,
                "descending": arxiv.SortOrder.Descending,
            }.get(self.sort_order, arxiv.SortOrder.Descending)
            
        except ImportError:
            raise ImportError(
                "Could not import arxiv python package. "
                "Please install it with `pip install arxiv`."
            )
        return self
    
    def search(
        self, 
        query: str,
        category: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: The search query
            category: Specific arXiv category to search in (e.g., 'cs.AI', 'physics')
            date_range: Optional dict with 'from_date' and 'to_date' keys
            
        Returns:
            List of paper metadata dictionaries with URLs
        """
        try:
            # Clean the query by removing problematic characters
            query = query[:self.max_query_length]
            
            # Add category filter if specified
            if category:
                query = f"cat:{category} AND {query}"
                
            # Create search object with sorting preferences
            search = self.arxiv_search(
                query=query,
                max_results=self.top_k_results,
                sort_by=self.arxiv_sort_criterion,
                sort_order=self.arxiv_sort_order,
            )
            
            # Execute the search
            results = list(search.results())
            
        except self.arxiv_exceptions as ex:
            logger.error(f"ArXiv API error: {ex}")
            return []
        
        # Process the results
        papers = []
        for result in results:
            # Get all available links (abstract page, PDF, etc.)
            links = {link.title if link.title else "unknown": link.href for link in result.links}
            
            # Basic metadata for all papers
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "published_date": str(result.published.date()),
                "updated_date": str(result.updated.date()),
                "arxiv_id": result.entry_id.split("/")[-1],  # Extract ID from URL
                "abstract_url": links.get("alternate", result.entry_id),
                "pdf_url": result.pdf_url,
                "categories": result.categories,
                "primary_category": result.primary_category,
            }
            
            # Add optional metadata if available
            if result.doi:
                paper["doi"] = result.doi
            if result.comment:
                paper["comment"] = result.comment
            if result.journal_ref:
                paper["journal_reference"] = result.journal_ref
                
            papers.append(paper)
            
        return papers
