import argparse
import asyncio
from pathlib import Path
from typing import List

# Use the actual Agent/Runner import path based on your project structure
# Assuming 'agents' is the correct package/module name for the SDK
from agents import Agent, Runner
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Define the models for our agents


class ReviewerProfile(BaseModel):
    """Profile of a potential reviewer."""

    name: str = Field(
        description="Descriptive title for the reviewer based on their specialty (e.g., 'Computational Linguist', 'Experimental Physicist')."
    )
    expertise: List[str] = Field(
        description="List of areas of expertise of the reviewer."
    )
    system_prompt: str = Field(
        description="System prompt that would be used for this reviewer agent."
    )
    rationale: str = Field(
        description="Rationale for selecting this reviewer for the paper."
    )


class ProposedReviewers(BaseModel):
    """Container for proposed reviewers."""

    reviewers: List[ReviewerProfile] = Field(description="List of proposed reviewers.")
    paper_summary: str = Field(description="Brief summary of the paper content.")
    key_topics: List[str] = Field(
        description="Key topics and domains covered in the paper."
    )


class ReviewerAssessment(BaseModel):
    """Assessment of proposed reviewers."""

    selected_reviewers: List[ReviewerProfile] = Field(
        description="List of selected reviewers."
    )
    selection_rationale: str = Field(
        description="Rationale for the final selection of reviewers."
    )
    diversity_analysis: str = Field(
        description="Analysis of the diversity of expertise in the selected reviewer group."
    )


# --- Agent Creation Functions ---


def create_reviewer_proposer_agent(model: str = "gpt-4o-mini") -> Agent:
    """Creates and returns the Reviewer Proposer Agent."""
    return Agent(
        name="ReviewerProposerAgent",
        instructions="""
You are an expert agent responsible for analyzing scientific papers and proposing appropriate reviewers.

Your task is to:
1. Analyze the content of a scientific paper provided as text input
2. Identify key topics, methodologies, and domains of expertise covered
3. Propose at least 4 distinct reviewer profiles with diverse expertise that would be well-suited to review this paper
4. For each reviewer, create a detailed system prompt that could guide their review of the paper

When creating reviewer profiles:
- Consider domain expertise needed to properly assess the paper's methods and claims
- Ensure diversity of perspectives (different specialties, theoretical frameworks, methodologies)
- Consider both technical expertise and broader contextual knowledge
- Create distinctive profiles with different backgrounds and specializations
- The 'name' field should be a descriptive title reflecting the reviewer's specialty (e.g., 'Computational Linguist', 'Experimental Physicist'), not a person's name.
- Design system prompts that would guide each reviewer to focus on specific aspects of the paper

Your output should include:
- A brief summary of the paper
- Key topics identified
- At least 4 reviewer profiles, each with:
  - A descriptive specialty title (as 'name')
  - Areas of expertise
  - A detailed system prompt that could be used to guide their review
  - Rationale for why this reviewer profile is appropriate for this paper
        """,
        output_type=ProposedReviewers,
        model=model,
    )


def create_reviewer_selector_agent(model: str = "gpt-4o-mini") -> Agent:
    """Creates and returns the Reviewer Selector Agent."""
    return Agent(
        name="ReviewerSelectorAgent",
        instructions="""
You are an expert agent responsible for assessing proposed reviewers and selecting the final set of reviewers for a scientific paper.

Your input will be the JSON representation of the proposed reviewers.

Your task is to:
1. Review the list of proposed reviewer profiles (identified by their specialty title)
2. Assess each reviewer's suitability for reviewing the paper based on expertise and perspective
3. Select at least 2 (but preferably 3-4) reviewer profiles that collectively offer the best coverage of expertise
4. Ensure the selected reviewers provide diverse perspectives and complementary expertise
5. Provide a rationale for your selection

When selecting reviewers, consider:
- Relevance of expertise to the paper's specific content and methodologies
- Balance of technical knowledge and broader contextual understanding
- Diversity of perspectives to ensure comprehensive review
- Complementary expertise across the selected reviewer group
- Avoiding redundancy in reviewer expertise

Your output should include:
- The list of selected reviewer profiles (including name, expertise, system_prompt, rationale)
- A detailed rationale for your selection
- An analysis of how the selected reviewers collectively provide diverse perspectives
        """,
        output_type=ReviewerAssessment,  # Outputting the full assessment including rationale
        model=model,
    )


# --- Main Execution Block (for standalone testing) ---


async def run_standalone_reviewer_flow(paper_content: str):
    """Demonstrates running the proposer and selector agents directly."""

    # 1. Create and run the Proposer Agent
    logger.info("Creating and running Reviewer Proposer Agent...")
    proposer_agent = create_reviewer_proposer_agent()
    proposed_result = await Runner.run(proposer_agent, paper_content)

    # Try to cast the output, handle failure
    try:
        if not proposed_result:
            raise ValueError("Proposer agent returned None")
        proposed_reviewers = proposed_result.final_output_as(ProposedReviewers)
        if not proposed_reviewers or not proposed_reviewers.reviewers:
            raise ValueError("Proposer agent returned empty or invalid reviewer list")
    except Exception as e:
        logger.error(
            f"Reviewer Proposer Agent failed or produced invalid output: {e}. Result: {proposed_result}"
        )
        return None

    logger.info(f"Proposed {len(proposed_reviewers.reviewers)} reviewers.")

    # Log proposed reviewers
    logger.info("\n--- Proposed Reviewers ---")
    for r in proposed_reviewers.reviewers:
        logger.info(
            f"  Name: {r.name}, Expertise: {r.expertise}, Rationale: {r.rationale}"
        )
    logger.info("-" * 20)

    # 2. Create and run the Selector Agent
    logger.info("Creating and running Reviewer Selector Agent...")
    selector_agent = create_reviewer_selector_agent()
    selection_input = proposed_reviewers.model_dump_json()
    selection_result = await Runner.run(selector_agent, selection_input)

    # Try to cast the output, handle failure
    try:
        if not selection_result:
            raise ValueError("Selector agent returned None")
        reviewer_selection = selection_result.final_output_as(ReviewerAssessment)
        if not reviewer_selection or not reviewer_selection.selected_reviewers:
            raise ValueError(
                "Selector agent returned empty or invalid selected reviewer list"
            )
    except Exception as e:
        logger.error(
            f"Reviewer Selector Agent failed or produced invalid output: {e}. Result: {selection_result}"
        )
        return None

    logger.info(f"Selected {len(reviewer_selection.selected_reviewers)} reviewers.")
    logger.info(f"Selection Rationale: {reviewer_selection.selection_rationale}")
    logger.info(f"Diversity Analysis: {reviewer_selection.diversity_analysis}")

    # 3. Extract the final dictionary (Name -> System Prompt)
    selected_reviewers_dict = {
        reviewer.name: reviewer.system_prompt
        for reviewer in reviewer_selection.selected_reviewers
    }

    # Log selected reviewers
    logger.info("\n--- Final Selected Reviewers (Name: Prompt) ---")
    for name, prompt in selected_reviewers_dict.items():
        logger.info(
            f"  {name}: Prompt length = {len(prompt)}"
        )  # Log length instead of full prompt
    logger.info("-" * 20)

    return selected_reviewers_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Reviewer Finder agent flow on a specified paper file."
    )
    parser.add_argument(
        "paper_file",
        type=str,
        help="Path to the markdown file of the scientific paper.",
    )
    args = parser.parse_args()

    # Load paper content from file for standalone execution
    try:
        paper_content = Path(args.paper_file).read_text(encoding="utf-8")
        logger.info(f"Loaded paper content from: {args.paper_file}")
    except Exception as e:
        logger.error(f"Failed to load paper file {args.paper_file}: {e}")
        paper_content = None

    if paper_content:
        # Initialize client for standalone execution
        asyncio.run(run_standalone_reviewer_flow(paper_content))
    else:
        logger.info("Exiting due to paper loading failure.")
