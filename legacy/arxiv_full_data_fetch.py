"""
ArXiv API functions for searching articles and downloading PDFs asynchronously.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

import aiohttp
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ArxivAPI(BaseModel):
    """
    API client for arXiv to search papers and download PDFs.
    Uses direct HTTP requests instead of the arxiv library.
    """
    max_results: int = Field(default=10, description="Maximum number of search results to return")
    sort_by: str = Field(default="relevance", description="Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')")
    sort_order: str = Field(default="descending", description="Sort direction ('ascending' or 'descending')")
    
    # arXiv API constants
    ARXIV_API_URL: str = "http://export.arxiv.org/api/query"
    ARXIV_NAMESPACE: Dict[str, str] = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    
    arxiv_sort_by: str = Field(default="relevance", exclude=True)
    arxiv_sort_order: str = Field(default="descending", exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        self.arxiv_sort_by = {
            "relevance": "relevance", 
            "lastUpdatedDate": "lastUpdatedDate", 
            "submittedDate": "submittedDate"
        }.get(self.sort_by, "relevance")
        
        self.arxiv_sort_order = {
            "ascending": "ascending", 
            "descending": "descending"
        }.get(self.sort_order, "descending")

    
    async def search_articles(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search arXiv using a list of search terms.
        
        Args:
            search_terms: List of search terms/keywords
            category: Optional arXiv category to filter by (e.g., 'cs.AI', 'physics')
            
        Returns:
            List of paper metadata dictionaries
        """
        # Join search terms with OR operators for broader results
        query = " OR ".join(search_terms)
    
            
        # Construct query parameters
        params = {
            'search_query': query,
            'start': 0,
            'max_results': self.max_results,
            'sortBy': self.arxiv_sort_by,
            'sortOrder': self.arxiv_sort_order
        }
        
        try:
            # Make async HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ARXIV_API_URL, params=params) as response:
                    response.raise_for_status()
                    xml_data = await response.text()
                    
                    # Parse XML response
                    return self._parse_arxiv_response(xml_data)
                
        except aiohttp.ClientError as ex:
            logger.error(f"ArXiv API error: {ex}")
            return []
    
    def _parse_arxiv_response(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        Parse the XML response from arXiv API.
        
        Args:
            xml_data: XML response from arXiv API
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Extract entries (papers)
            entries = root.findall(".//atom:entry", self.ARXIV_NAMESPACE)
            
            for entry in entries:
                # Skip the first entry if it's the OpenSearch entry
                if entry.find(".//atom:title", self.ARXIV_NAMESPACE).text == "ArXiv Query:":
                    continue
                    
                paper = {}
                
                # Extract basic metadata
                paper["title"] = entry.find(".//atom:title", self.ARXIV_NAMESPACE).text.strip()
                paper["abstract"] = entry.find(".//atom:summary", self.ARXIV_NAMESPACE).text.strip()
                paper["arxiv_id"] = entry.find(".//atom:id", self.ARXIV_NAMESPACE).text.split("/abs/")[-1]
                
                # Extract authors
                authors = entry.findall(".//atom:author/atom:name", self.ARXIV_NAMESPACE)
                paper["authors"] = [author.text for author in authors]
                
                # Extract links
                links = entry.findall(".//atom:link", self.ARXIV_NAMESPACE)
                paper["abstract_url"] = ""
                paper["pdf_url"] = ""
                
                for link in links:
                    href = link.get("href", "")
                    rel = link.get("rel", "")
                    title = link.get("title", "")
                    
                    if rel == "alternate":
                        paper["abstract_url"] = href
                    elif title == "pdf" or href.endswith(".pdf"):
                        paper["pdf_url"] = href
                
                # Extract dates
                published = entry.find(".//atom:published", self.ARXIV_NAMESPACE).text
                updated = entry.find(".//atom:updated", self.ARXIV_NAMESPACE).text
                
                # Convert dates to YYYY-MM-DD format
                paper["published_date"] = datetime.fromisoformat(published.replace("Z", "+00:00")).date().isoformat()
                paper["updated_date"] = datetime.fromisoformat(updated.replace("Z", "+00:00")).date().isoformat()
                
                # Extract categories
                categories = entry.findall(".//arxiv:primary_category", self.ARXIV_NAMESPACE)
                primary_category = categories[0].get("term") if categories else ""
                
                all_categories = entry.findall(".//atom:category", self.ARXIV_NAMESPACE)
                category_terms = [cat.get("term") for cat in all_categories]
                
                paper["primary_category"] = primary_category
                paper["categories"] = category_terms
                
                # Extract optional fields
                journal_ref = entry.find(".//arxiv:journal_ref", self.ARXIV_NAMESPACE)
                if journal_ref is not None:
                    paper["journal_reference"] = journal_ref.text
                    
                doi = entry.find(".//arxiv:doi", self.ARXIV_NAMESPACE)
                if doi is not None:
                    paper["doi"] = doi.text
                    
                comment = entry.find(".//arxiv:comment", self.ARXIV_NAMESPACE)
                if comment is not None:
                    paper["comment"] = comment.text
                
                papers.append(paper)
                
        except ET.ParseError as ex:
            logger.error(f"Error parsing XML response: {ex}")
        
        return [paper["pdf_url"] for paper in papers]
    
    async def fetch_pdf(self, article_url: str, output_dir: Optional[str] = None) -> str:
        """
        Download a PDF from an arXiv article URL asynchronously.
        
        Args:
            article_url: URL to the arXiv article (can be abstract page or direct PDF URL)
            output_dir: Directory to save the PDF (if None, uses a temporary directory)
            
        Returns:
            Path to the downloaded PDF file
        """
        # Convert abstract URL to PDF URL if needed
        if not article_url.endswith('.pdf'):
            # Extract article ID
            if 'arxiv.org/abs/' in article_url:
                article_id = article_url.split('arxiv.org/abs/')[-1]
            elif 'arxiv.org/pdf/' in article_url:
                article_id = article_url.split('arxiv.org/pdf/')[-1].replace('.pdf', '')
            else:
                raise ValueError(f"Could not extract arXiv ID from URL: {article_url}")
                
            # Convert to PDF URL
            pdf_url = f"https://arxiv.org/pdf/{article_id}.pdf"
        else:
            pdf_url = article_url
        
        # Create temporary directory if output_dir is not specified
        if output_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = temp_dir.name
        
        # Extract filename from URL
        filename = os.path.basename(pdf_url)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Download the PDF asynchronously
            logger.info(f"Downloading PDF from {pdf_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    response.raise_for_status()
                    
                    # Read response in chunks and write to file
                    with open(output_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                    
            logger.info(f"PDF downloaded to {output_path}")
            return output_path
            
        except aiohttp.ClientError as e:
            logger.error(f"Error downloading PDF: {e}")
            raise
