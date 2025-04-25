import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, RunContextWrapper
from openai import OpenAI
from typing import Optional, List
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Paper file path
PAPER_PATH = "data/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC.md"

class PaperAnalysisResult(BaseModel):
    """Result of paper analysis."""
    paper_type: str
    focus_areas: List[str]
    recommended_sections: List[str]
    main_topics: List[str]
    estimated_complexity: int  # 1-10 scale

@dataclass
class PaperValidatorContext:
    """Context for paper validation."""
    paper_path: str = PAPER_PATH
    paper_content: Optional[str] = None  # Direct markdown content
    expected_sections: list[str] = None
    min_score_to_proceed: int = 5
    grammar_check_enabled: bool = True
    paper_type: str = "research"  # Options: "research", "review", "case_study", "thesis"
    focus_areas: Optional[list[str]] = None  # e.g. ["methodology", "results", "discussion"]
    auto_detect_paper_properties: bool = True  # Whether to auto-detect paper type and focus areas
    paper_analysis: Optional[PaperAnalysisResult] = None
    
    def __post_init__(self):
        # If paper_content is not provided but paper_path is, load from file
        if self.paper_content is None and self.paper_path:
            try:
                with open(self.paper_path, 'r', encoding='utf-8') as file:
                    self.paper_content = file.read()
            except Exception as e:
                print(f"Warning: Failed to read paper from path: {str(e)}")
        
        if self.expected_sections is None:
            # Default expected sections for a research paper
            if self.paper_type == "research":
                self.expected_sections = [
                    "Abstract",
                    "Introduction",
                    "Literature Review/Background",
                    "Methodology",
                    "Results",
                    "Discussion",
                    "Conclusion",
                    "References"
                ]
            # For a review paper
            elif self.paper_type == "review":
                self.expected_sections = [
                    "Abstract",
                    "Introduction",
                    "Methods (Search Strategy)",
                    "Literature Overview",
                    "Discussion/Analysis",
                    "Conclusion",
                    "References"
                ]
            # For a case study
            elif self.paper_type == "case_study":
                self.expected_sections = [
                    "Abstract",
                    "Introduction",
                    "Case Presentation",
                    "Discussion",
                    "Conclusion",
                    "References"
                ]
            # For a thesis
            elif self.paper_type == "thesis":
                self.expected_sections = [
                    "Abstract",
                    "Introduction",
                    "Literature Review",
                    "Methodology",
                    "Results",
                    "Discussion",
                    "Conclusion",
                    "References",
                    "Appendices"
                ]
            else:
                # Default if paper type is not recognized
                self.expected_sections = [
                    "Abstract",
                    "Introduction",
                    "Main Body",
                    "Conclusion",
                    "References"
                ]
        
        if self.focus_areas is None:
            self.focus_areas = []

@function_tool
def analyze_paper_properties(wrapper: RunContextWrapper[PaperValidatorContext]) -> str:
    """
    Analyzes a paper to determine its type, focus areas, and other properties.
    
    Args:
        wrapper: Context wrapper containing paper content.
        
    Returns:
        str: JSON string of paper properties analysis.
    """
    paper_content = wrapper.context.paper_content
    
    # Check if paper content is available
    if not paper_content:
        return "Error: No paper content available to analyze."
    
    # Use OpenAI to analyze paper properties
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an academic paper analyzer. Extract key information about the paper's type, structure, and focus areas."
            },
            {
                "role": "user",
                "content": f"Analyze this academic paper and determine: 1) The paper type (research, review, case_study, thesis, or other), 2) The main focus areas or key sections that should receive special attention, 3) The recommended sections for this type of paper, 4) The main topics or themes covered, 5) The estimated complexity on a scale of 1-10. Format your response as structured JSON.\n\nPaper content:\n{paper_content}"
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=500
    )
    
    # Extract and structure the results
    analysis_json = response.choices[0].message.content
    
    # Update the context with the analysis results
    try:
        result = PaperAnalysisResult.model_validate_json(analysis_json)
        wrapper.context.paper_analysis = result
        wrapper.context.paper_type = result.paper_type
        wrapper.context.focus_areas = result.focus_areas
        
        # Update expected sections if available from analysis
        if result.recommended_sections:
            wrapper.context.expected_sections = result.recommended_sections
    except Exception as e:
        # If parsing fails, return the error but don't block the process
        return f"Paper analysis completed but error parsing results: {str(e)}. Continuing with default settings.\n\nRaw analysis: {analysis_json}"
    
    return f"Paper successfully analyzed as '{result.paper_type}' with focus areas: {', '.join(result.focus_areas)}. Complexity level: {result.estimated_complexity}/10."

def dynamic_instructions(wrapper: RunContextWrapper[PaperValidatorContext], agent: Agent[PaperValidatorContext]) -> str:
    """
    Dynamically generate instructions based on context parameters.
    
    Args:
        wrapper: Context wrapper containing validation parameters
        agent: The agent using these instructions
        
    Returns:
        str: Customized instructions for the agent
    """
    ctx = wrapper.context
    paper_type_desc = {
        "research": "empirical research study with original data collection and analysis",
        "review": "literature review synthesizing existing research",
        "case_study": "detailed analysis of a specific case or instance",
        "thesis": "comprehensive academic thesis or dissertation"
    }.get(ctx.paper_type, "scientific paper")
    
    focus_instructions = ""
    if ctx.focus_areas:
        focus_instructions = f" Pay special attention to the following areas: {', '.join(ctx.focus_areas)}."
    
    complexity_note = ""
    if ctx.paper_analysis and hasattr(ctx.paper_analysis, 'estimated_complexity'):
        complexity = ctx.paper_analysis.estimated_complexity
        if complexity > 7:
            complexity_note = f" This is a complex paper (rated {complexity}/10), so be thorough in your analysis."
        elif complexity < 4:
            complexity_note = f" This is a relatively simple paper (rated {complexity}/10), so focus on core elements."
    
    auto_detect_instruction = ""
    if ctx.auto_detect_paper_properties:
        auto_detect_instruction = " First analyze the paper properties to determine its type and focus areas."
    
    return f"""You are a structure validator for a {paper_type_desc}.{auto_detect_instruction}

Your task is to analyze the paper's structure and check if it contains all expected sections: {', '.join(ctx.expected_sections)}.

{focus_instructions}{complexity_note}

The minimum quality score to proceed to in-depth review is {ctx.min_score_to_proceed}/10.

Provide only concise summaries and specific recommendations without detailed analysis. 
Use the generate_final_summary tool to create a concise final report combining both analyses and include a clear decision about whether the paper should proceed to in-depth review by another agent.
"""

@function_tool
def validate_structure(wrapper: RunContextWrapper[PaperValidatorContext]) -> str:
    """
    Validates the structure of a scientific paper.
    
    Args:
        wrapper: Context wrapper containing paper content and expected sections.
        
    Returns:
        str: A summary of the paper's structure and improvement recommendations.
    """
    paper_content = wrapper.context.paper_content
    expected_sections = wrapper.context.expected_sections
    paper_type = wrapper.context.paper_type
    
    # Check if paper content is available
    if not paper_content:
        return "Error: No paper content available to analyze."
    
    # Use OpenAI to analyze the paper structure
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a scientific paper structure validator for a {paper_type} paper. Analyze the given paper and identify which of the expected sections are present and which are missing. Provide only a brief summary and specific improvement recommendations without detailed analysis."
            },
            {
                "role": "user",
                "content": f"Analyze the structure of this scientific paper and identify which of these expected sections are present: {', '.join(expected_sections)}. Then provide ONLY: 1) A brief summary of present/missing sections, 2) Specific improvement recommendations. Be concise.\n\nIMPORTANT: DO NOT include any summary of the paper's content, subject matter, or findings. Focus EXCLUSIVELY on structural elements like section headers and organizational aspects.\n\nPaper content:\n{paper_content}"
            }
        ],
        temperature=0.0,
        max_tokens=500
    )
    
    validation_result = response.choices[0].message.content
    return validation_result

@function_tool
def check_grammar_punctuation(wrapper: RunContextWrapper[PaperValidatorContext]) -> str:
    """
    Checks grammar and punctuation in a scientific paper.
    
    Args:
        wrapper: Context wrapper containing the paper content and check options.
        
    Returns:
        str: A summary of grammar issues and improvement recommendations.
    """
    paper_content = wrapper.context.paper_content
    
    # Skip if grammar check is disabled
    if not wrapper.context.grammar_check_enabled:
        return "Grammar check skipped as per configuration."
    
    # Check if paper content is available
    if not paper_content:
        return "Error: No paper content available to analyze."
    
    # Use OpenAI to analyze grammar and punctuation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a scientific writing expert focused on grammar and punctuation. Your task is to analyze the provided scientific paper and provide only a concise summary of grammar and punctuation issues and specific improvement recommendations."
            },
            {
                "role": "user",
                "content": f"Review this scientific paper for grammar and punctuation issues. Provide ONLY: 1) A concise summary of overall writing quality, 2) 3-5 specific improvement recommendations with brief examples. Do not provide a detailed section-by-section analysis.\n\nIMPORTANT: DO NOT include any summary of the paper's content, subject matter, or findings. Focus EXCLUSIVELY on writing quality, grammar, syntax, and punctuation.\n\nPaper content:\n{paper_content}"
            }
        ],
        temperature=0.0,
        max_tokens=500
    )
    
    grammar_analysis = response.choices[0].message.content
    return grammar_analysis

@function_tool
def generate_final_summary(wrapper: RunContextWrapper[PaperValidatorContext], structure_analysis: str, grammar_analysis: str) -> str:
    """
    Generates a final concise summary combining structure and grammar analyses with a decision.
    
    Args:
        wrapper: Context wrapper containing validation criteria.
        structure_analysis (str): The structure analysis of the paper.
        grammar_analysis (str): The grammar and punctuation analysis of the paper.
        
    Returns:
        str: A concise final summary with improvement recommendations and review decision.
    """
    min_score = wrapper.context.min_score_to_proceed
    paper_type = wrapper.context.paper_type
    
    # Add paper analysis information if available
    paper_info = ""
    if wrapper.context.paper_analysis:
        analysis = wrapper.context.paper_analysis
        paper_info = f"This is a {paper_type} paper with an estimated complexity of {analysis.estimated_complexity}/10."
    
    # Use OpenAI to create a concise combined summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a scientific paper reviewer evaluating a {paper_type} paper. {paper_info} Provide concise and actionable feedback. Create a very brief evaluation focused ONLY on structure and grammar issues - DO NOT summarize the paper's content or topic. Make a clear decision about whether the paper should proceed to in-depth review. When evaluating, give higher weight to scientific content and essential structure than to minor formatting or grammar issues."
            },
            {
                "role": "user",
                "content": f"Create a concise final validation report that combines these analyses. DO NOT include a description of what the paper is about or summarize its content - focus ONLY on structure and grammar evaluation.\n\nInclude:\n1) 'Structure Assessment' section with brief evaluation of structural elements only (not paper content)\n2) 'Writing Quality' section with brief evaluation of grammar and writing\n3) 'Key Recommendations' section with 3-5 actionable structure/grammar improvements\n4) 'Review Decision' section with a clear YES or NO decision on whether the paper should proceed to in-depth review\n\nThe decision should be based on whether the quality meets the minimum standard of {min_score}/10. Note that scientific content and essential structure should count more toward the evaluation than minor grammar or formatting issues.\n\nIMPORTANT: DO NOT include phrases like 'The paper provides...' or 'This study explores...' or ANY description of the paper's content, subject matter, or findings. Focus EXCLUSIVELY on structural elements and writing quality.\n\nStructure Analysis:\n{structure_analysis}\n\nGrammar Analysis:\n{grammar_analysis}"
            }
        ],
        temperature=0.0,
        max_tokens=500
    )
    
    final_summary = response.choices[0].message.content
    return final_summary

# Create the structure validator agent for export
structure_validator_agent = Agent[PaperValidatorContext](
    name="Structure Validator",
    instructions=dynamic_instructions,
    model="gpt-4o-mini",
    tools=[analyze_paper_properties, validate_structure, check_grammar_punctuation, generate_final_summary],
)

async def run_validator(paper_content=None, paper_path=PAPER_PATH, paper_type=None, min_score=5, focus_areas=None, grammar_check=True, auto_detect=True):
    """Run the structure validator on a paper.
    
    Args:
        paper_content (str, optional): The markdown content of the paper to validate
        paper_path (str, optional): Path to the paper to validate (used only if paper_content is not provided)
        paper_type (str, optional): Type of paper - "research", "review", "case_study", or "thesis". If None and auto_detect=True, will be detected automatically.
        min_score (int): Minimum score needed to proceed to review (1-10)
        focus_areas (list, optional): Specific areas to focus on during validation. If None and auto_detect=True, will be detected automatically.
        grammar_check (bool): Whether to check grammar
        auto_detect (bool): Whether to automatically detect paper properties
        
    Returns:
        str: The final validation output
    """
    # Create context
    context = PaperValidatorContext(
        paper_content=paper_content,
        paper_path=paper_path if paper_content is None else None,
        paper_type=paper_type if paper_type else "research",  # Will be overridden if auto_detect is True
        min_score_to_proceed=min_score,
        focus_areas=focus_areas,
        grammar_check_enabled=grammar_check,
        auto_detect_paper_properties=auto_detect
    )
    
    result = await Runner.run(
        structure_validator_agent, 
        input="Please analyze the paper. Start by examining the paper's properties, then check its structure and grammar, and finally provide a concise summary report with the most important findings, recommendations, and a clear decision whether the paper should proceed to in-depth review.",
        context=context
    )
    
    return result.final_output

async def main():
    """Main function for running as a standalone script."""
    
    # Example using direct markdown content
    sample_markdown = """ """
    
    output2 = await run_validator(
        paper_content=sample_markdown,
        auto_detect=True,
        min_score=5
    )
    print("\nResult from direct markdown input:")
    print(output2)
    

if __name__ == "__main__":
    asyncio.run(main())