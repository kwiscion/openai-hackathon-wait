import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from Agents import Agent, Runner
from loguru import logger
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DecisionType(str, Enum):
    ACCEPT = "accept"
    MINOR_REVISIONS = "minor revisions"
    MAJOR_REVISIONS = "major revisions"
    REJECT = "reject"
    REJECT_AND_RESUBMIT = "reject and resubmit"


class PublicationDecision(BaseModel):
    decision: DecisionType = Field(description="The final decision on the publication.")
    rationale: str = Field(description="Comprehensive rationale explaining the decision.")
    key_strengths: str = Field(description="Key strengths that influenced the decision.")
    key_weaknesses: str = Field(description="Key weaknesses that influenced the decision.")
    specific_revisions: Optional[str] = Field(
        description="Specific revisions required (if minor/major revisions or resubmit).", 
        default=None
    )
    priority_issues: Optional[List[str]] = Field(
        description="List of priority issues that must be addressed in revision.",
        default=None
    )
    ethical_considerations: Optional[str] = Field(
        description="Ethical considerations that affected the decision.",
        default=None
    )


DECISION_PROMPT = """
You are an academic journal editor making a final decision on manuscript submissions.

Your task is to synthesize all available information about a manuscript and render a clear, fair, and justified publication decision.

INPUT INFORMATION:
1. A synthesized review document combining perspectives from multiple peer reviewers
2. The original manuscript text (in markdown format)
3. Literature search context (when available)
4. Technical analysis regarding structure and language (when available)

DECISION CATEGORIES:
- Accept As Is: Publishable in current form with no changes needed
- Minor Revisions: Fundamentally sound but requires limited, specific changes
- Major Revisions: Has potential merit but requires substantial changes
- Reject: Unsuitable due to fatal flaws, lack of novelty/significance, poor scope fit, or ethical issues
- Reject & Resubmit: Rejected in current form, but a substantially revised version may be considered

DECISION FACTORS TO CONSIDER:
1. Methodological soundness and scientific validity
2. Originality and contribution to the field
3. Clarity and quality of presentation
4. Ethical considerations
5. Alignment with journal scope/standards
6. Reviewer consensus and confidence levels
7. Feasibility of addressing identified weaknesses

DECISION STRUCTURE:
- Provide a clear, definitive decision
- Include comprehensive rationale explaining your decision
- Highlight key strengths and weaknesses
- If revisions are recommended, provide specific guidance on required changes
- Prioritize issues that must be addressed
- Address any ethical considerations
- Be fair, constructive, and specific

Your decision should represent a careful synthesis of all input information, applying your best editorial judgment to determine the most appropriate outcome for this manuscript.
"""


# Create the agent definition
publication_decision_agent = Agent(
    name="PublicationDecisionAgent",
    instructions=DECISION_PROMPT,
    model="gpt-4o",
    output_type=PublicationDecision,
)


class DecisionOrchestratorInput(BaseModel):
    """Input structure for the Decision Orchestrator."""
    synthesized_review_path: str
    manuscript_path: str
    literature_context: Optional[str] = None
    technical_analysis: Optional[str] = None


class PublicationDecisionOrchestrator:
    """
    Orchestrates the publication decision process by combining synthesized reviews,
    manuscript content, and optional contextual information.
    """
    
    def __init__(self, synthesized_review_path: str, manuscript_path: str, 
                literature_context_path: Optional[str] = None, 
                technical_analysis_path: Optional[str] = None):
        """Initialize the publication decision orchestrator with paths to input files."""
        self.synthesized_review_path = synthesized_review_path
        self.manuscript_path = manuscript_path
        self.literature_context_path = literature_context_path
        self.technical_analysis_path = technical_analysis_path
        
    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}
    
    def _load_text(self, file_path: str) -> str:
        """Load text from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""
    
    def _prepare_decision_prompt(
        self, 
        synthesized_review: Dict[str, Any], 
        manuscript: str,
        literature_context: Optional[str] = None,
        technical_analysis: Optional[str] = None
    ) -> str:
        """
        Prepare a comprehensive prompt for the decision agent.
        
        Args:
            synthesized_review: The synthesized review data
            manuscript: The manuscript text
            literature_context: Optional context from literature search
            technical_analysis: Optional analysis of structure and language
            
        Returns:
            str: Formatted prompt for the decision agent
        """
        prompt_parts = []
        
        # Add synthesized review
        prompt_parts.append("## SYNTHESIZED REVIEW")
        prompt_parts.append(json.dumps(synthesized_review, indent=2))
        prompt_parts.append("")
        
        # Add manuscript
        prompt_parts.append("## MANUSCRIPT")
        prompt_parts.append(manuscript)
        prompt_parts.append("")
        
        # Add literature context if available
        if literature_context:
            prompt_parts.append("## LITERATURE CONTEXT")
            prompt_parts.append(literature_context)
            prompt_parts.append("")
        else:
            prompt_parts.append("## LITERATURE CONTEXT")
            prompt_parts.append("No literature context provided.")
            prompt_parts.append("")
        
        # Add technical analysis if available
        if technical_analysis:
            prompt_parts.append("## TECHNICAL ANALYSIS")
            prompt_parts.append(technical_analysis)
            prompt_parts.append("")
        else:
            prompt_parts.append("## TECHNICAL ANALYSIS")
            prompt_parts.append("No technical analysis provided.")
            prompt_parts.append("")
            
        # Add instructions for the decision
        prompt_parts.append("## TASK")
        prompt_parts.append(
            "Based on all the information above, please provide a final publication decision "
            "including a clear recommendation (accept, minor revisions, major revisions, reject, "
            "or reject and resubmit) and a detailed rationale for your decision."
        )
        
        return "\n".join(prompt_parts)
    
    async def make_decision(self) -> Optional[PublicationDecision]:
        """Process all inputs and generate a publication decision."""
        logger.info("Starting publication decision process...")
        
        # Load the synthesized review
        synthesized_review = self._load_json(self.synthesized_review_path)
        if not synthesized_review:
            logger.error(f"Could not load synthesized review from {self.synthesized_review_path}")
            return None
        
        # Load the manuscript
        manuscript = self._load_text(self.manuscript_path)
        if not manuscript:
            logger.error(f"Could not load manuscript from {self.manuscript_path}")
            return None
        
        # Load optional context if provided
        literature_context = None
        if self.literature_context_path:
            literature_context = self._load_text(self.literature_context_path)
            
        technical_analysis = None
        if self.technical_analysis_path:
            technical_analysis = self._load_text(self.technical_analysis_path)
        
        # Prepare the prompt with all available information
        prompt = self._prepare_decision_prompt(
            synthesized_review, 
            manuscript, 
            literature_context,
            technical_analysis
        )
        
        # Run the decision agent
        logger.info("Running publication decision agent...")
        result = await Runner.run(publication_decision_agent, prompt)
        
        # Return the decision
        return result.final_output
    
    def save_decision(self, decision: PublicationDecision, output_path: Optional[str] = None):
        """Save the publication decision to a JSON file."""
        if output_path is None:
            # Get manuscript filename without extension
            manuscript_path = Path(self.manuscript_path)
            manuscript_name = manuscript_path.stem
            
            # Create decisions directory if it doesn't exist
            decisions_dir = Path("data/decisions")
            decisions_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output path
            output_path = str(decisions_dir / f"{manuscript_name}_decision.json")
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(decision.model_dump(), f, indent=2)
            logger.info(f"Decision saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving decision to {output_path}: {e}")


async def main():
    """Main function to run the orchestrator from command line."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate publication decision based on synthesized reviews")
    parser.add_argument("--review",
                      help="Path to the synthesized review JSON file")
    parser.add_argument("--manuscript", 
                      help="Path to the manuscript markdown file")
    parser.add_argument("--literature", help="Path to literature context file (optional)")
    parser.add_argument("--technical", help="Path to technical analysis file (optional)")
    parser.add_argument("--output", help="Path for output decision JSON file (optional, defaults to data/decisions/MANUSCRIPT_NAME_decision.json)")
    
    args = parser.parse_args()
    
    # Initialize the orchestrator with provided paths
    orchestrator = PublicationDecisionOrchestrator(
        synthesized_review_path=args.review,
        manuscript_path=args.manuscript,
        literature_context_path=args.literature,
        technical_analysis_path=args.technical
    )
    
    # Generate the decision
    decision = await orchestrator.make_decision()
    
    if decision:
        # Save the decision
        orchestrator.save_decision(decision, args.output)
        
        # Print summary to console
        logger.info("\n=== PUBLICATION DECISION ===")
        logger.info(f"DECISION: {decision.decision.upper()}")
        logger.info(f"\nRATIONALE:\n{decision.rationale}")
        logger.info(f"\nKEY STRENGTHS:\n{decision.key_strengths}")
        logger.info(f"\nKEY WEAKNESSES:\n{decision.key_weaknesses}")
        
        # Print revision requirements if applicable
        if decision.decision in [DecisionType.MINOR_REVISIONS, DecisionType.MAJOR_REVISIONS, DecisionType.REJECT_AND_RESUBMIT]:
            logger.info(f"\nREQUIRED REVISIONS:\n{decision.specific_revisions or 'None specified'}")
            
            if decision.priority_issues:
                priority_issues = "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(decision.priority_issues))
                logger.info(f"\nPRIORITY ISSUES:\n{priority_issues}")
        
        # Print ethical considerations if present
        if decision.ethical_considerations:
            logger.info(f"\nETHICAL CONSIDERATIONS:\n{decision.ethical_considerations}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 