import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from agents import Agent, RunContextWrapper, Runner, function_tool
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()


class FigureAnalysisResult(BaseModel):
    """Result of figure analysis."""

    figure_type: str
    main_message: str
    key_elements: List[str]


@dataclass
class FigureExtractorContext:
    """Context for figure extraction and analysis."""

    paper_path: Optional[str] = None
    figure_paths: Optional[List[str]] = None
    figures_content: Optional[Dict[str, bytes]] = (
        None  # Figure filename -> binary content
    )
    paper_content: Optional[str] = None  # Direct markdown content of the paper
    paper_title: Optional[str] = None
    paper_abstract: Optional[str] = None
    detailed_analysis: bool = True
    extract_data_points: bool = False
    figure_analysis_results: Optional[Dict[str, FigureAnalysisResult]] = None

    def __post_init__(self):
        # Initialize figure_analysis_results as empty dict if None
        if self.figure_analysis_results is None:
            self.figure_analysis_results = {}

        # If paper_content is not provided but paper_path is, load from file
        if self.paper_content is None and self.paper_path:
            try:
                with open(self.paper_path, "r", encoding="utf-8") as file:
                    self.paper_content = file.read()
            except Exception as e:
                print(f"Warning: Failed to read paper from path: {str(e)}")

        # Load figures if paths are provided but content isn't
        if self.figures_content is None and self.figure_paths:
            self.figures_content = {}
            for fig_path in self.figure_paths:
                try:
                    with open(fig_path, "rb") as file:
                        self.figures_content[os.path.basename(fig_path)] = file.read()
                except Exception as e:
                    print(
                        f"Warning: Failed to read figure from path {fig_path}: {str(e)}"
                    )


def dynamic_instructions(
    wrapper: RunContextWrapper[FigureExtractorContext],
    agent: Agent[FigureExtractorContext],
) -> str:
    """
    Dynamically generate instructions based on context parameters.

    Args:
        wrapper: Context wrapper containing analysis parameters
        agent: The agent using these instructions

    Returns:
        str: Customized instructions for the agent
    """
    ctx = wrapper.context

    detail_level = "detailed" if ctx.detailed_analysis else "high-level"
    data_points_instruction = (
        "Also extract specific data points from charts and graphs."
        if ctx.extract_data_points
        else ""
    )

    return f"""You are a figure analysis expert for academic papers.

Your task is to extract and interpret the context, meaning, and significance of figures in research papers. 
Provide {detail_level} analysis of each figure, interpreting what they show, their relationship to the paper's findings, and their significance.
{data_points_instruction}

For each figure:
1. Identify the figure type (graph, chart, diagram, microscopy image, etc.)
2. Explain what the figure is showing and its main message
3. Point out key elements and features
4. Interpret the figure's significance in the context of the paper
5. Note any limitations or potential alternative interpretations

Use the appropriate tools to analyze each figure individually, then generate a comprehensive summary using the generate_figures_summary tool.
"""


@function_tool
def extract_paper_metadata(wrapper: RunContextWrapper[FigureExtractorContext]) -> str:
    """
    Extracts the title and abstract from the paper content to provide context for figure analysis.

    Args:
        wrapper: Context wrapper containing paper content.

    Returns:
        str: Confirmation of metadata extraction.
    """
    paper_content = wrapper.context.paper_content

    # Check if paper content is available
    if not paper_content:
        return "Error: No paper content available to extract metadata."

    # Use OpenAI to extract title and abstract
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an academic paper analyzer. Extract the title and abstract from the paper content.",
            },
            {
                "role": "user",
                "content": f"Extract the title and abstract from this academic paper. Return a JSON object with two fields: 'title' and 'abstract'.\n\nPaper content:\n{paper_content[:10000]}",  # Limit to first 10k chars
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    # Extract and structure the results
    result_json = response.choices[0].message.content

    # Update the context with the extracted metadata
    try:
        import json

        result = json.loads(result_json)
        wrapper.context.paper_title = result.get("title")
        wrapper.context.paper_abstract = result.get("abstract")
    except Exception as e:
        return f"Metadata extraction completed but error parsing results: {str(e)}. Continuing with available information."

    return f"Successfully extracted paper metadata: '{wrapper.context.paper_title}'"


@function_tool
def analyze_figure(
    wrapper: RunContextWrapper[FigureExtractorContext], figure_path: str
) -> str:
    """
    Analyzes a single figure from a paper.

    Args:
        wrapper: Context wrapper containing analysis parameters.
        figure_path: Path or key to the figure to analyze.

    Returns:
        str: Analysis of the figure.
    """
    figures_content = wrapper.context.figures_content
    paper_title = wrapper.context.paper_title
    paper_abstract = wrapper.context.paper_abstract
    detailed = wrapper.context.detailed_analysis
    extract_data = wrapper.context.extract_data_points

    # Check if figure content is available
    if not figures_content or figure_path not in figures_content:
        return f"Error: Figure '{figure_path}' not found in available figures."

    # Prepare the image for the API
    figure_data = figures_content[figure_path]

    # Encode image to base64
    base64_image = base64.b64encode(figure_data).decode("utf-8")

    # Prepare context for analysis
    context_text = f"Paper Title: {paper_title}" if paper_title else ""
    if paper_abstract:
        context_text += f"\nPaper Abstract: {paper_abstract}"

    detail_instruction = (
        "Provide a detailed analysis" if detailed else "Provide a concise overview"
    )
    data_point_instruction = (
        "Extract and list specific data points visible in charts/graphs."
        if extract_data
        else ""
    )

    # Use OpenAI to analyze the figure
    response = client.chat.completions.create(
        model="gpt-4o",  # Using vision model
        messages=[
            {
                "role": "system",
                "content": f"You are a scientific figure analysis expert. {detail_instruction} of academic figures, explaining what they show, their significance, and key features. {data_point_instruction} Return your analysis in a clear, structured format with appropriate headings.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this figure from an academic paper.\n\nContext: {context_text}\n\nPlease structure your response with the following sections:\n1. Figure Type\n2. Main Message\n3. Key Elements\n4. Significance\n5. Limitations/Alternative Interpretations",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=1000,
    )

    analysis_result = response.choices[0].message.content

    # Store a structured version of the analysis
    try:
        # Extract figure type and main message using a second API call
        structure_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured information from this figure analysis.",
                },
                {
                    "role": "user",
                    "content": f"From the following figure analysis, extract: 1) the figure type, 2) the main message, and 3) key elements as a list. Return as JSON with fields 'figure_type', 'main_message', and 'key_elements'.\n\n{analysis_result}",
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=500,
        )

        structured_data = structure_response.choices[0].message.content
        import json

        structured_result = json.loads(structured_data)

        # Store the structured result
        wrapper.context.figure_analysis_results[figure_path] = FigureAnalysisResult(
            figure_type=structured_result.get("figure_type", "Unknown"),
            main_message=structured_result.get("main_message", ""),
            key_elements=structured_result.get("key_elements", []),
        )
    except Exception as e:
        print(f"Error storing structured figure analysis: {str(e)}")

    return analysis_result


@function_tool
def extract_data_from_chart(
    wrapper: RunContextWrapper[FigureExtractorContext], figure_path: str
) -> str:
    """
    Extracts numerical data points from charts and graphs.

    Args:
        wrapper: Context wrapper containing the figures.
        figure_path: Path or key to the figure to analyze.

    Returns:
        str: Extracted data points in tabular format.
    """
    figures_content = wrapper.context.figures_content

    # Check if figure content is available
    if not figures_content or figure_path not in figures_content:
        return f"Error: Figure '{figure_path}' not found in available figures."

    # Skip if the figure is not suitable for data extraction
    figure_results = wrapper.context.figure_analysis_results
    if figure_path in figure_results:
        figure_type = figure_results[figure_path].figure_type.lower()
        if not any(
            keyword in figure_type
            for keyword in ["graph", "chart", "plot", "histogram"]
        ):
            return f"This figure ({figure_path}) does not appear to contain extractable data points (type: {figure_type})."

    # Prepare the image for the API
    figure_data = figures_content[figure_path]
    base64_image = base64.b64encode(figure_data).decode("utf-8")

    # Use OpenAI to extract data points
    response = client.chat.completions.create(
        model="gpt-4o",  # Using vision model
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction specialist. Extract numerical data from charts, graphs, and plots in scientific figures. Provide the data in a clean, tabular format as best as possible.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all numerical data points from this figure. If the figure is a graph or chart, try to provide the data in a table format. Include units where applicable. If exact values are difficult to determine, provide best estimates and indicate this. If the figure doesn't contain extractable data points, state this clearly.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        max_tokens=800,
    )

    data_extraction = response.choices[0].message.content
    return data_extraction


@function_tool
def detect_figure_relationships(
    wrapper: RunContextWrapper[FigureExtractorContext],
) -> str:
    """
    Detects relationships between figures and how they complement each other.

    Args:
        wrapper: Context wrapper containing analyzed figures.

    Returns:
        str: Analysis of figure relationships.
    """
    figure_results = wrapper.context.figure_analysis_results

    # Check if we have analyzed figures
    if not figure_results or len(figure_results) < 2:
        return "Insufficient figures to analyze relationships. At least 2 analyzed figures are required."

    # Prepare summarized figure information
    figures_info = []
    for fig_path, analysis in figure_results.items():
        figures_info.append(
            {
                "figure_id": fig_path,
                "figure_type": analysis.figure_type,
                "main_message": analysis.main_message,
                "key_elements": analysis.key_elements,
            }
        )

    import json

    figures_json = json.dumps(figures_info)

    # Use OpenAI to analyze relationships
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a scientific figure analyst. Analyze relationships between figures in an academic paper and explain how they complement or contrast with each other.",
            },
            {
                "role": "user",
                "content": f"Analyze the relationships between these figures from an academic paper. Explain how they complement or contrast with each other, potential narrative flow, and how they collectively contribute to the paper's findings. Be specific when referring to figures by using their IDs.\n\nFigures information:\n{figures_json}",
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )

    relationship_analysis = response.choices[0].message.content
    return relationship_analysis


@function_tool
def generate_figures_summary(wrapper: RunContextWrapper[FigureExtractorContext]) -> str:
    """
    Generates a comprehensive summary of all analyzed figures and their relevance to the paper.

    Args:
        wrapper: Context wrapper containing analysis results.

    Returns:
        str: Comprehensive summary of figures.
    """
    figure_results = wrapper.context.figure_analysis_results
    paper_title = wrapper.context.paper_title
    paper_abstract = wrapper.context.paper_abstract

    # Check if we have analyzed figures
    if not figure_results:
        return "No analyzed figures available to summarize."

    # Prepare summarized figure information
    figures_info = []
    for fig_path, analysis in figure_results.items():
        figures_info.append(
            {
                "figure_id": fig_path,
                "figure_type": analysis.figure_type,
                "main_message": analysis.main_message,
                "key_elements": analysis.key_elements,
            }
        )

    import json

    figures_json = json.dumps(figures_info)

    # Context information
    context_info = ""
    if paper_title:
        context_info += f"Paper Title: {paper_title}\n"
    if paper_abstract:
        context_info += f"Paper Abstract: {paper_abstract}\n"

    # Use OpenAI to generate a comprehensive summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a scientific figure analysis expert. Create a comprehensive yet concise summary of all analyzed figures from a paper, explaining their collective significance and insights.",
            },
            {
                "role": "user",
                "content": f"Generate a comprehensive summary of all the analyzed figures from this paper. Explain what insights they collectively provide, how they support the paper's arguments or findings, and any patterns or themes across the figures.\n\nContext:\n{context_info}\n\nFigures information:\n{figures_json}",
            },
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    summary = response.choices[0].message.content
    return summary


@function_tool
def convert_figures_to_text(wrapper: RunContextWrapper[FigureExtractorContext]) -> str:
    """
    Converts visual figures into textual descriptions for accessibility.

    Args:
        wrapper: Context wrapper containing the figures.

    Returns:
        str: Accessible text descriptions of all figures.
    """
    figures_content = wrapper.context.figures_content

    # Check if figure content is available
    if not figures_content:
        return "Error: No figures available to convert to text."

    # Store all accessibility descriptions
    all_descriptions = []

    # Process each figure
    for fig_path, figure_data in figures_content.items():
        # Encode image to base64
        base64_image = base64.b64encode(figure_data).decode("utf-8")

        # Use OpenAI to create accessibility description
        response = client.chat.completions.create(
            model="gpt-4o",  # Using vision model
            messages=[
                {
                    "role": "system",
                    "content": "You are an accessibility specialist. Create detailed text descriptions of scientific figures for vision-impaired readers. Be thorough but concise, focusing on the key visual elements, data trends, and important features.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Create an accessibility text description for this scientific figure. The description should enable a vision-impaired reader to understand what the figure shows, including key data, trends, and visual elements. Make your description thorough but concise.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
        )

        description = response.choices[0].message.content
        all_descriptions.append(f"Figure: {fig_path}\n{description}\n")

    return "\n".join(all_descriptions)


# Create the figure extractor agent for export
figure_extractor_agent = Agent[FigureExtractorContext](
    name="Figure Context Extractor",
    instructions=dynamic_instructions,
    model="gpt-4o-mini",
    tools=[
        extract_paper_metadata,
        analyze_figure,
        extract_data_from_chart,
        detect_figure_relationships,
        generate_figures_summary,
        convert_figures_to_text,
    ],
)


async def run_figure_extractor(
    figure_paths=None,
    figures_content=None,
    paper_path=None,
    paper_content=None,
    detailed_analysis=True,
    extract_data_points=False,
):
    """Run the figure context extractor on a set of figures.

    Args:
        figure_paths (List[str], optional): Paths to the figure files to analyze
        figures_content (Dict[str, bytes], optional): Dict mapping figure names to binary content
        paper_path (str, optional): Path to the paper containing the figures
        paper_content (str, optional): The markdown content of the paper
        detailed_analysis (bool): Whether to perform detailed analysis (vs high-level)
        extract_data_points (bool): Whether to extract numerical data points from charts/graphs

    Returns:
        str: The final analysis output
    """
    # Create context
    context = FigureExtractorContext(
        paper_path=paper_path,
        figure_paths=figure_paths,
        figures_content=figures_content,
        paper_content=paper_content,
        detailed_analysis=detailed_analysis,
        extract_data_points=extract_data_points,
    )

    result = await Runner.run(
        figure_extractor_agent,
        input="Please analyze the figures from this paper. Start by extracting paper metadata if available, then analyze each figure individually, detect relationships between figures, and finally provide a comprehensive summary of all figures.",
        context=context,
    )

    return result.final_output


async def main():
    """Main function for running as a standalone script."""

    # Example figure paths - replace with real paths for testing
    figure_paths = [
        "data/Bagdasarian_et_al_(2024)_Acute Effects_of_Hallucinogens_on_FC/_page_0_Figure_5.jpeg",
        "data/Bagdasarian_et_al_(2024)_Acute Effects_of_Hallucinogens_on_FC/_page_1_Figure_3.jpeg",
    ]

    # Example paper path
    paper_path = "data/Bagdasarian_et_al_(2024)_Acute Effects_of_Hallucinogens_on_FC/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC.md"

    # Run figure extraction - comment out if not testing
    try:
        output = await run_figure_extractor(
            figure_paths=figure_paths,
            paper_path=paper_path,
            detailed_analysis=True,
            extract_data_points=True,
        )
        print("\nFigure Analysis Result:")
        print(output)
    except Exception as e:
        print(f"Error running figure extractor: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
