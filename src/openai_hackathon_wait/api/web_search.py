import os
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

class WebSearchResult(BaseModel):
    """Model for web search results"""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    content: str = Field(..., description="Snippet or content of the search result")

class WebSearchResponse(BaseModel):
    """Model for web search response"""
    results: List[WebSearchResult] = Field([], description="List of search results")
    query: str = Field(..., description="Original search query")

def perform_web_search(query: str, max_results: int = 5) -> WebSearchResponse:
    """
    Perform a web search using OpenAI's search capability
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        WebSearchResponse: Search results and metadata
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        # Create a chat completion with the web search tool enabled
        response = client.chat.completions.create(
            model="gpt-4o",  # You can change to other models that support web tools
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Search the web for information and provide factual, detailed responses with citations."},
                {"role": "user", "content": f"Search the web for: {query}"}
            ],
            tools=[{"type": "web_search"}],
            tool_choice={"type": "web_search"}
        )
        
        # Process the tool results from the response
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                tool_calls = choice.message.tool_calls
                
                if tool_calls:
                    # Get the search results from the first tool call
                    search_results_raw = tool_calls[0].function.arguments
                    
                    # Parse the search results (this may vary depending on the actual format)
                    # This is a placeholder for how you might process the results
                    search_results = []
                    for result in search_results_raw.get("results", [])[:max_results]:
                        search_results.append(
                            WebSearchResult(
                                title=result.get("title", ""),
                                url=result.get("url", ""),
                                content=result.get("content", "")
                            )
                        )
                    
                    return WebSearchResponse(
                        results=search_results,
                        query=query
                    )
        
        # Fallback to using the content response if tool parsing fails
        return WebSearchResponse(
            results=[
                WebSearchResult(
                    title="Search Result",
                    url="",
                    content=response.choices[0].message.content if response.choices else "No results found"
                )
            ],
            query=query
        )
                    
    except Exception as e:
        # Handle any exceptions
        print(f"Error performing web search: {str(e)}")
        return WebSearchResponse(
            results=[],
            query=query
        )

def deep_research(topic: str, depth: int = 3, max_results_per_query: int = 3) -> Dict[str, Any]:
    """
    Perform deep research on a topic by recursively searching for information
    
    Args:
        topic: The main research topic
        depth: How many levels of research to perform (default: 3)
        max_results_per_query: Maximum number of results per query (default: 3)
        
    Returns:
        Dict: Research results organized hierarchically
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Initial search on the main topic
    main_results = perform_web_search(topic, max_results_per_query)
    
    # Organize the research results
    research = {
        "topic": topic,
        "results": [
            {
                "title": result.title,
                "url": result.url,
                "content": result.content,
                "sub_topics": []
            }
            for result in main_results.results
        ]
    }
    
    # Perform deeper research if depth > 1
    if depth > 1:
        for i, result in enumerate(research["results"]):
            # Generate sub-topics based on this result
            prompt = f"Based on this information: '{result['content']}', generate 2-3 specific sub-topics or questions that would be valuable to research further about '{topic}'."
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a research assistant helping to identify important sub-topics for deeper investigation."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            sub_topics_text = response.choices[0].message.content
            
            # Extract sub-topics from the text
            # This is a simple approach - in a real implementation, you might want to use more sophisticated parsing
            sub_topics = [line.strip() for line in sub_topics_text.split("\n") if line.strip()]
            sub_topics = [topic.split(". ", 1)[-1] if ". " in topic else topic for topic in sub_topics]
            sub_topics = [topic.strip('"').strip("'").strip() for topic in sub_topics]
            
            # Limit to first 2-3 sub-topics
            sub_topics = sub_topics[:min(3, len(sub_topics))]
            
            # Research each sub-topic if we're not at the maximum depth
            if depth > 2:
                for sub_topic in sub_topics:
                    sub_results = perform_web_search(f"{topic} {sub_topic}", max_results_per_query)
                    
                    result["sub_topics"].append({
                        "topic": sub_topic,
                        "results": [
                            {
                                "title": sub_result.title,
                                "url": sub_result.url,
                                "content": sub_result.content
                            }
                            for sub_result in sub_results.results
                        ]
                    })
            else:
                # Just add the sub-topics without researching them
                result["sub_topics"] = [{"topic": sub_topic, "results": []} for sub_topic in sub_topics]
    
    return research

def format_research_results(research: Dict[str, Any]) -> str:
    """
    Format research results into a readable text format
    
    Args:
        research: Research results from deep_research
        
    Returns:
        str: Formatted research report
    """
    report = [f"# Research Report: {research['topic']}\n"]
    
    for i, result in enumerate(research["results"]):
        report.append(f"## Source {i+1}: {result['title']}")
        report.append(f"URL: {result['url']}")
        report.append(f"Summary: {result['content']}\n")
        
        if result["sub_topics"]:
            report.append("### Key Sub-topics:")
            
            for j, sub_topic in enumerate(result["sub_topics"]):
                report.append(f"#### {j+1}. {sub_topic['topic']}")
                
                if sub_topic["results"]:
                    for k, sub_result in enumerate(sub_topic["results"]):
                        report.append(f"* {sub_result['title']}")
                        report.append(f"  URL: {sub_result['url']}")
                        report.append(f"  Summary: {sub_result['content']}\n")
    
    return "\n".join(report) 