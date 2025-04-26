import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from loguru import logger
from pydantic import BaseModel, Field
from agents import Agent, Runner
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Ensure OpenAI client is initialized
client = OpenAI()


# Define the models for our agents

class ReviewerProfile(BaseModel):
    """Profile of a potential reviewer."""
    name: str = Field(description="Descriptive title for the reviewer based on their specialty (e.g., 'Computational Linguist', 'Experimental Physicist').")
    expertise: List[str] = Field(description="List of areas of expertise of the reviewer.")
    system_prompt: str = Field(description="System prompt that would be used for this reviewer agent.")
    rationale: str = Field(description="Rationale for selecting this reviewer for the paper.")

class ProposedReviewers(BaseModel):
    """Container for proposed reviewers."""
    reviewers: List[ReviewerProfile] = Field(description="List of proposed reviewers.")
    paper_summary: str = Field(description="Brief summary of the paper content.")
    key_topics: List[str] = Field(description="Key topics and domains covered in the paper.")

class ReviewerAssessment(BaseModel):
    """Assessment of proposed reviewers."""
    selected_reviewers: List[ReviewerProfile] = Field(description="List of selected reviewers.")
    selection_rationale: str = Field(description="Rationale for the final selection of reviewers.")
    diversity_analysis: str = Field(description="Analysis of the diversity of expertise in the selected reviewer group.")

# Create the agents

reviewer_proposer_agent = Agent(
    name="ReviewerProposerAgent",
    instructions="""
You are an expert agent responsible for analyzing scientific papers and proposing appropriate reviewers.

Your task is to:
1. Analyze the content of a scientific paper provided in markdown format
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
    model="gpt-4o",
)

reviewer_selector_agent = Agent(
    name="ReviewerSelectorAgent",
    instructions="""
You are an expert agent responsible for assessing proposed reviewers and selecting the final set of reviewers for a scientific paper.

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
- A dictionary of selected reviewers with the specialty title as key and system message as value
- A detailed rationale for your selection
- An analysis of how the selected reviewers collectively provide diverse perspectives
    """,
    output_type=ReviewerAssessment,
    model="gpt-4o",
)

class ReviewerFinder:
    def __init__(self, paper_path: str):
        """Initialize the reviewer finder with path to the paper file."""
        self.paper_path = paper_path
        
    def load_paper(self) -> str:
        """Load paper content from file."""
        try:
            with open(self.paper_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading paper: {e}")
            return ""
            
    async def find_reviewers(self) -> Optional[Dict[str, str]]:
        """Find appropriate reviewers for the paper."""
        paper_content = self.load_paper()
        if not paper_content:
            logger.warning("No paper content found or error loading paper.")
            return None
        
        # Step 1: Propose reviewers
        logger.info("Analyzing paper and proposing reviewers...")
        proposed_result = await Runner.run(reviewer_proposer_agent, paper_content)
        proposed_reviewers = proposed_result.final_output_as(ProposedReviewers)
        
        if not proposed_reviewers:
            logger.error("Failed to propose reviewers.")
            return None
            
        logger.info(f"Proposed {len(proposed_reviewers.reviewers)} reviewers")
        
        # Log details of each proposed reviewer
        logger.info("\n=== PROPOSED REVIEWERS ===")
        for reviewer in proposed_reviewers.reviewers:
            logger.info(f"PROPOSED REVIEWER ROLE: {reviewer.name}")
            logger.info(f"  Expertise: {', '.join(reviewer.expertise)}")
            logger.info(f"  Rationale: {reviewer.rationale}")
            # Optionally log the system prompt too, though it might be long
            # logger.info(f"  System Prompt: {reviewer.system_prompt}") 
            logger.info("---")
        
        # Step 2: Select final reviewers
        logger.info("\nSelecting final reviewers...")
        selection_input = json.dumps(proposed_reviewers.model_dump())
        selection_result = await Runner.run(reviewer_selector_agent, selection_input)
        reviewer_selection = selection_result.final_output_as(ReviewerAssessment)
        
        if not reviewer_selection:
            logger.error("Failed to select reviewers.")
            return None
            
        logger.info(f"Selected {len(reviewer_selection.selected_reviewers)} reviewers")
        logger.info(f"Selection rationale: {reviewer_selection.selection_rationale}")
        
        # Convert list of ReviewerProfile objects to Dict[str, str]
        selected_reviewers_dict = {
            reviewer.name: reviewer.system_prompt 
            for reviewer in reviewer_selection.selected_reviewers
        }
        
        return selected_reviewers_dict
        
    def save_reviewers(self, reviewers: Dict[str, str], output_path: Optional[str] = None):
        """Save the selected reviewers to a JSON file."""
        if output_path is None:
            # Create output filename based on input filename
            input_path = Path(self.paper_path)
            output_path = str(input_path.parent / f"{input_path.stem}_reviewer_selection.json")
            
        with open(output_path, 'w') as f:
            json.dump(reviewers, f, indent=4)
            
        logger.info(f"Selected reviewers saved to {output_path}")

async def main(paper_file: str):
    # Initialize and run the reviewer finder
    finder = ReviewerFinder(paper_file)
    reviewers = await finder.find_reviewers()
    
    if reviewers:
        # Save the selected reviewers
        finder.save_reviewers(reviewers)
        
        # Log the selected reviewers
        logger.info("\n=== SELECTED REVIEWERS ===")
        for name, system_prompt in reviewers.items():
            logger.info(f"REVIEWER ROLE: {name}")
            logger.info(f"SYSTEM PROMPT: {system_prompt}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Reviewer Finder on a specified paper.")
    parser.add_argument("paper_file", type=str, help="Path to the markdown file of the scientific paper.")
    args = parser.parse_args()
    
    asyncio.run(main(args.paper_file))
