"""Util that calls Arxiv."""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator
from agents import function_tool


logger = logging.getLogger(__name__)


class ArxivAPIWrapper(BaseModel):
    """Wrapper around ArxivAPI.

    To use, you should have the ``arxiv`` python package installed.
    https://lukasschwab.me/arxiv.py/index.html
    This wrapper will use the Arxiv API to conduct searches and
    fetch document summaries. By default, it will return the document summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    Set doc_content_chars_max=None if you don't want to limit the content size.

    Attributes:
        top_k_results: number of the top-scored document used for the arxiv tool
        ARXIV_MAX_QUERY_LENGTH: the cut limit on the query used for the arxiv tool.
        load_max_docs: a limit to the number of loaded documents
        load_all_available_meta:
            if True: the `metadata` of the loaded Documents contains all available
            meta info (see https://lukasschwab.me/arxiv.py/index.html#Result),
            if False: the `metadata` contains only the published date, title,
            authors and summary.
        doc_content_chars_max: an optional cut limit for the length of a document's
            content

    Example:
        .. code-block:: python

            from langchain.utilities.arxiv import ArxivAPIWrapper
            arxiv = ArxivAPIWrapper(
                top_k_results = 3,
                ARXIV_MAX_QUERY_LENGTH = 300,
                load_max_docs = 3,
                load_all_available_meta = False,
                doc_content_chars_max = 40000
            )
            arxiv.run("tree of thought llm)
    """

    arxiv_search: Any = None  #: :meta private:
    arxiv_exceptions: Any = "not_found"  #: :meta private:
    top_k_results: int = 3  # Reduced to 3 for practical content extraction
    ARXIV_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = 100
    load_all_available_meta: bool = False
    doc_content_chars_max: Optional[int] = 8000  # Increased to get more content
    mindate: str = None
    maxdate: str = None

    @model_validator(mode="after")
    def validate_environment(self) -> "ArxivAPIWrapper":
        try:
            import arxiv

            self.arxiv_search = arxiv.Search
            self.arxiv_exceptions = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            )
        except ImportError:
            raise ImportError(
                "Could not import arxiv python package. "
                "Please install it with `pip install arxiv`."
            )
        return self

    def run(self, query: str) -> str:
        """
        Performs an arxiv search and A single string
        with the publish date, title, authors, and summary
        for each article separated by two newlines.

        If an error occurs or no documents found, error text
        is returned instead. Wrapper for
        https://lukasschwab.me/arxiv.py/index.html#Search

        Args:
            query: a plaintext search query
        """  # noqa: E501
        try:
            query = query[: self.ARXIV_MAX_QUERY_LENGTH]
            results = self.arxiv_search(  # type: ignore
                query, max_results=self.top_k_results
            ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            f"Published: {result.updated.date()}\n"
            f"Title: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

    def get_paper_content(self, result) -> str:
        """
        Downloads a paper PDF and extracts its text content.

        Args:
            result: An arxiv search result object

        Returns:
            The extracted text content from the PDF
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(
                "PyMuPDF not installed. Install with 'pip install pymupdf' to extract paper content."
            )
            return "Paper content extraction not available. Install PyMuPDF with 'pip install pymupdf'."

        try:
            # Create a temporary directory to store the downloaded PDF
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Generate a filename in the temporary directory
                filename = os.path.join(tmpdirname, "paper.pdf")

                # Download the PDF
                pdf_url = result.pdf_url
                logger.info(f"Downloading PDF from {pdf_url}")

                # Use the download_pdf method from the result object
                doc_file_name = result.download_pdf(dirpath=tmpdirname)

                # Open the PDF with PyMuPDF
                with fitz.open(doc_file_name) as doc_file:
                    # Extract text from the first few pages (to avoid returning too much content)
                    max_pages = min(5, len(doc_file))
                    text = ""
                    for i in range(max_pages):
                        text += doc_file[i].get_text()

                    # Limit the length of the text
                    if self.doc_content_chars_max:
                        text = text[: self.doc_content_chars_max]

                    return text
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {e}")
            return f"Error extracting paper content: {str(e)}"

    def load(self, query: str, include_content: bool = False) -> List[Dict[str, Any]]:
        """
        Run Arxiv search and get the article meta information with optional full text content.
        See https://lukasschwab.me/arxiv.py/index.html#Search

        Returns: a list of paper metadata dicts with relevant information

        Performs an arxiv search and returns metadata about the papers.

        Args:
            query: a plaintext search query
            include_content: whether to include full paper content (may be slow)
        """
        try:
            # Remove the ":" and "-" from the query, as they can cause search problems
            query = query.replace(":", "").replace("-", "")
            results = self.arxiv_search(
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
            ).results()
        except self.arxiv_exceptions as ex:
            logger.debug("Error on arxiv: %s", ex)
            return []

        papers = []
        for result in results:
            if self.load_all_available_meta:
                extra_metadata = {
                    "entry_id": result.entry_id,
                    "published_first_time": str(result.published.date()),
                    "comment": result.comment,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "links": [link.href for link in result.links],
                }
            else:
                extra_metadata = {}

            paper = {
                "published": str(result.updated.date()),
                "title": result.title,
                "authors": ", ".join(a.name for a in result.authors),
                "summary": result.summary,
                "url": result.pdf_url,
                **extra_metadata,
            }

            # If requested, extract and include the paper content
            if include_content:
                paper["content"] = self.get_paper_content(result)

            papers.append(paper)

        return papers


# Initialize ArXiv wrapper
arxiv_wrapper = ArxivAPIWrapper()


@function_tool
def arxiv_tool(query: str) -> Dict[str, Any]:
    """
    Searches the arXiv API for scientific papers related to a given query.

    Useful for answering questions related to physics, computer science, mathematics,
    quantitative biology, quantitative finance, statistics, and other academic topics
    based on scientific preprints from arXiv.

    Args:
        query: A search query related to scientific research.
        include_content: Whether to include the full text content of the papers (may be slow).

    Returns:
        A dictionary with information about the retrieved papers, optionally including their content.
    """
    include_content = False
    if include_content:
        # Use the load method with content extraction
        papers = arxiv_wrapper.load(query, include_content=True)
        if not papers:
            return {
                "error": f"No good arXiv results found for '{query}'.",
                "papers": [],
            }
        return {"papers": papers}
    else:
        # Use the simpler run method without content extraction
        result = arxiv_wrapper.run(query)

        if "No good Arxiv Result was found" in result or "Arxiv exception" in result:
            return {
                "error": f"No good arXiv results found for '{query}'.",
                "papers": [],
            }

        # Parse the string format from run() into a structured format
        papers = []
        if result:
            paper_texts = result.split("\n\n")
            for paper_text in paper_texts:
                lines = paper_text.strip().split("\n")
                paper = {}
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        paper[key.lower()] = value
                if paper:
                    papers.append(paper)

        return {"papers": papers}
