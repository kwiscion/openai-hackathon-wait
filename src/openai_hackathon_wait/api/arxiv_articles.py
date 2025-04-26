"""
ArXiv API functions for searching articles and downloading PDFs asynchronously.
"""

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from pydantic import BaseModel, Field

from openai_hackathon_wait.utils.get_text_from_pdf import get_text_from_pdf_with_pymupdf

logger = logging.getLogger(__name__)


class ArxivAPI(BaseModel):
    """
    API client for arXiv to search papers and download PDFs.
    Uses direct HTTP requests instead of the arxiv library.
    """

    max_results: int = Field(
        default=10, description="Maximum number of search results to return"
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')",
    )
    sort_order: str = Field(
        default="descending", description="Sort direction ('ascending' or 'descending')"
    )

    # arXiv API constants
    ARXIV_API_URL: str = "http://export.arxiv.org/api/query"
    ARXIV_NAMESPACE: Dict[str, str] = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    arxiv_sort_by: str = Field(default="relevance", exclude=True)
    arxiv_sort_order: str = Field(default="descending", exclude=True)

    def __init__(self, **data):
        super().__init__(**data)

        self.arxiv_sort_by = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }.get(self.sort_by, "relevance")

        self.arxiv_sort_order = {
            "ascending": "ascending",
            "descending": "descending",
        }.get(self.sort_order, "descending")

    async def search_articles(self, search_terms: List[str]) -> List[str]:
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
            "search_query": query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": self.arxiv_sort_by,
            "sortOrder": self.arxiv_sort_order,
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

            paper = {}

            # Extract entries (papers)
            entries = root.findall(".//atom:entry", self.ARXIV_NAMESPACE)

            for entry in entries:
                # Skip the first entry if it's the OpenSearch entry
                if (
                    entry.find(".//atom:title", self.ARXIV_NAMESPACE).text
                    == "ArXiv Query:"
                ):
                    continue

                # Extract links
                links = entry.findall(".//atom:link", self.ARXIV_NAMESPACE)

                for link in links:
                    href = link.get("href", "")
                    rel = link.get("rel", "")
                    title = link.get("title", "")

                    if title == "pdf" or href.endswith(".pdf"):
                        paper["pdf_url"] = href

                # Extract dates
                published = entry.find(".//atom:published", self.ARXIV_NAMESPACE).text
                updated = entry.find(".//atom:updated", self.ARXIV_NAMESPACE).text

                # Convert dates to YYYY-MM-DD format
                paper["published_date"] = (
                    datetime.fromisoformat(published.replace("Z", "+00:00"))
                    .date()
                    .isoformat()
                )
                paper["updated_date"] = (
                    datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    .date()
                    .isoformat()
                )

                # Extract categories
                categories = entry.findall(
                    ".//arxiv:primary_category", self.ARXIV_NAMESPACE
                )
                primary_category = categories[0].get("term") if categories else ""

                all_categories = entry.findall(".//atom:category", self.ARXIV_NAMESPACE)
                category_terms = [cat.get("term") for cat in all_categories]

                paper["primary_category"] = primary_category
                paper["categories"] = category_terms

                papers.append(paper)

        except ET.ParseError as ex:
            logger.error(f"Error parsing XML response: {ex}")

        return [paper["pdf_url"] for paper in papers]

    async def fetch_pdf_and_get_article_content(self, article_url: str) -> str:
        """
        Download a PDF from an arXiv article URL, extract its text content, and remove the temporary file.

        Args:
            article_url: URL to the arXiv article (can be abstract page or direct PDF URL)

        Returns:
            Extracted text content from the PDF
        """
        # Convert abstract URL to PDF URL if needed
        if not article_url.endswith(".pdf"):
            # Extract article ID
            if "arxiv.org/abs/" in article_url:
                article_id = article_url.split("arxiv.org/abs/")[-1]
            elif "arxiv.org/pdf/" in article_url:
                article_id = article_url.split("arxiv.org/pdf/")[-1].replace(".pdf", "")
            else:
                raise ValueError(f"Could not extract arXiv ID from URL: {article_url}")

            # Convert to PDF URL
            pdf_url = f"https://arxiv.org/pdf/{article_id}.pdf"
        else:
            pdf_url = article_url

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract filename from URL
            filename = os.path.basename(pdf_url)
            output_path = os.path.join(temp_dir, filename)

            try:
                # Download the PDF asynchronously
                logger.info(f"Downloading PDF from {pdf_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.get(pdf_url) as response:
                        response.raise_for_status()

                        # Read response in chunks and write to file
                        with open(output_path, "wb", encoding="utf-8") as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)

                logger.info(f"PDF downloaded to temporary file {output_path}")

                # Extract text from PDF
                text_content = await get_text_from_pdf_with_pymupdf(
                    pdf_path=output_path
                )

                logger.info(
                    f"Successfully extracted text from PDF ({len(text_content)} characters)"
                )
                return text_content

            except aiohttp.ClientError as e:
                logger.error(f"Error downloading PDF: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                raise
