import time
import urllib.error
import urllib.request
import os

import json
import time
import urllib.parse
from typing import Any, Dict, Iterator, List
from typing import Optional
import requests

from Bio import Entrez
from langchain_core.documents import Document
from pydantic import BaseModel, model_validator
from loguru import logger

Entrez.email = "your.email@example.com"


class PubMedAPIWrapper(BaseModel):
    """
    Wrapper around PubMed API.

    This wrapper will use the PubMed API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    Parameters:
        top_k_results: number of the top-scored document used for the PubMed tool
        MAX_QUERY_LENGTH: maximum length of the query.
          Default is 300 characters.
        doc_content_chars_max: maximum length of the document content.
          Content will be truncated if it exceeds this length.
          Default is 2000 characters.
        max_retry: maximum number of retries for a request. Default is 5.
        sleep_time: time to wait between retries.
          Default is 0.2 seconds.
        email: email address to be used for the PubMed API.
        api_key: API key to be used for the PubMed API.
    """

    parse: Any  #: :meta private:

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    max_retry: int = 5
    sleep_time: float = 0.2

    # Default values for the parameters
    top_k_results: int = 3
    MAX_QUERY_LENGTH: int = 300
    doc_content_chars_max: int = 2000
    email: str = "your_email@example.com"
    api_key: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the python package exists in environment."""
        try:
            import xmltodict

            values["parse"] = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict python package. "
                "Please install it with `pip install xmltodict`."
            )
        return values

    def run(self, query: str) -> str:
        """
        Run PubMed search and get the article meta information.
        See https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        It uses only the most informative fields of article meta information.
        """

        try:
            # Retrieve the top-k results for the query
            docs = [
                f"Published: {result['Published']}\n"
                f"Title: {result['Title']}\n"
                f"Copyright Information: {result['Copyright Information']}\n"
                f"Summary::\n{result['Summary']}"
                for result in self.load(query[: self.MAX_QUERY_LENGTH])
            ]

            # Join the results and limit the character count
            return (
                "\n\n".join(docs)[: self.doc_content_chars_max]
                if docs
                else "No good PubMed Result was found"
            )
        except Exception as ex:
            return f"PubMed exception: {ex}"

    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """
        print(f"Searching for: {query}")
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str({urllib.parse.quote(query)})
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
        )
        if self.api_key != "":
            url += f"&api_key={self.api_key}"
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)

        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)

    def load(self, query: str) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        """
        return list(self.lazy_load(query))

    def _dict2document(self, doc: dict) -> Document:
        summary = doc.pop("Summary")
        return Document(page_content=summary, metadata=doc)

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        for d in self.lazy_load(query=query):
            yield self._dict2document(d)

    def load_docs(self, query: str) -> List[Document]:
        return list(self.lazy_load_docs(query=query))

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id="
            + uid
            + "&webenv="
            + webenv
        )
        if self.api_key != "":
            url += f"&api_key={self.api_key}"

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = [
            f"{txt['@Label']}: {txt['#text']}"
            for txt in abstract_text
            if "#text" in txt and "@Label" in txt
        ]
        summary = (
            "\n".join(summaries)
            if summaries
            else (
                abstract_text
                if isinstance(abstract_text, str)
                else (
                    "\n".join(str(value) for value in abstract_text.values())
                    if isinstance(abstract_text, dict)
                    else "No abstract available"
                )
            )
        )
        a_d = ar.get("ArticleDate", {})
        pub_date = "-".join(
            [
                a_d.get("Year", ""),
                a_d.get("Month", ""),
                a_d.get("Day", ""),
            ]
        )

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Copyright Information": ar.get("Abstract", {}).get(
                "CopyrightInformation", ""
            ),
            "Summary": summary,
        }

class PubMedAPIWrapperImproved(PubMedAPIWrapper):
    def retrieve_article(self, uid: str, webenv: str) -> dict:
        print(f"Retrieving article {uid}")
        pmcid = get_pmcid(uid)
        if pmcid:
            full_text = fetch_full_text(pmcid)
            if full_text:
                # Parse and return the full-text content
                return {
                    "uid": uid,
                    "Title": full_text['documents'][0]['passages'][0]['infons']['title'],
                    "Published": full_text['documents'][0]['passages'][0]['infons']['date'],
                    "Summary": full_text['documents'][0]['passages'][0]['text'],
                    "Copyright Information": full_text['documents'][0]['passages'][0]['infons'].get('copyright', '')
                }
        # Fallback to abstract
        abstract = fetch_abstract(uid)
        return {
            "uid": uid,
            "Title": "",  # You can parse title from the abstract if needed
            "Published": "",
            "Summary": abstract,
            "Copyright Information": ""
        }


def get_pmcid(pmid):
    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
    records = Entrez.read(handle)
    handle.close()
    try:
        pmcid = records[0]['LinkSetDb'][0]['Link'][0]['Id']
        return pmcid
    except (IndexError, KeyError):
        return None


def fetch_full_text(pmcid):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{pmcid}/unicode"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def fetch_abstract(pmid):
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
    abstract = handle.read()
    handle.close()
    return abstract




class PubMedAgentTool:
    """
    A PubMed search tool for use with OpenAI Agents SDK.
    
    This tool allows agents to search PubMed for biomedical literature
    and retrieve article summaries.
    
    Parameters:
        top_k_results: Number of top-scored documents to return (default: 3)
        api_key: API key to be used for the PubMed API
        email: Email address to be used for the PubMed API
    """
    
    def __init__(
        self,
        top_k_results: int = 3,
        api_key: Optional[str] = None,
        email: str = "your_email@example.com",
    ):
        self.api_wrapper = PubMedAPIWrapperImproved(
            top_k_results=top_k_results,
            api_key=api_key or os.getenv("PUBMED_API_KEY", ""),
            email=email,
        )
