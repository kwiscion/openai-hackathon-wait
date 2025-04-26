import os

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pydantic import BaseModel

load_dotenv()


class Source(BaseModel):
    url: str
    title: str
    description: str


class DeepResearchResult(BaseModel):
    final_analysis: str
    sources: list[Source]


firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


def on_activity(activity):
    print(f"[{activity['type']}] {activity['message']}")


def perform_deep_research(
    query: str, max_depth: int = 5, max_urls: int = 15
) -> DeepResearchResult:
    results = firecrawl.deep_research(
        query=query,
        max_depth=max_depth,
        time_limit=180,
        max_urls=max_urls,
        on_activity=on_activity,
    )
    data = results["data"]

    return DeepResearchResult(
        final_analysis=data["finalAnalysis"],
        sources=[
            Source(
                url=source["url"],
                title=source["title"],
                description=source["description"],
            )
            for source in data["sources"]
        ],
    )