import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import Agent, RunContextWrapper, Runner
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

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
    rationale: str = Field(
        description="Comprehensive rationale explaining the decision."
    )
    key_strengths: str = Field(
        description="Key strengths that influenced the decision."
    )
    key_weaknesses: str = Field(
        description="Key weaknesses that influenced the decision."
    )
    specific_revisions: Optional[str] = Field(
        description="Specific revisions required (if minor/major revisions or resubmit).",
        default=None,
    )
    priority_issues: Optional[List[str]] = Field(
        description="List of priority issues that must be addressed in revision.",
        default=None,
    )
    ethical_considerations: Optional[str] = Field(
        description="Ethical considerations that affected the decision.", default=None
    )


# --- Context Definition ---
class PublicationDecisionContext(BaseModel):
    """Context object for the Publication Decision Agent."""

    synthesized_review: Dict[str, Any]
    manuscript: str
    literature_context: Optional[str] = None
    technical_analysis: Optional[str] = None
    manuscript_filename: str  # Added to generate default output filename


# --- Dynamic Instructions ---
def dynamic_instructions(
    wrapper: RunContextWrapper[PublicationDecisionContext],
    agent: Agent[PublicationDecisionContext],
) -> str:
    """
    Generate dynamic instructions for the publication decision agent based on context.
    """
    ctx = wrapper.context
    prompt_parts = []

    prompt_parts.append(
        """
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
    )

    # Add synthesized review
    prompt_parts.append("## SYNTHESIZED REVIEW")
    prompt_parts.append(json.dumps(ctx.synthesized_review, indent=2))
    prompt_parts.append("")

    # Add manuscript
    prompt_parts.append("## MANUSCRIPT")
    prompt_parts.append(ctx.manuscript)
    prompt_parts.append("")

    # Add literature context if available
    prompt_parts.append("## LITERATURE CONTEXT")
    if ctx.literature_context:
        prompt_parts.append(ctx.literature_context)
    else:
        prompt_parts.append("No literature context provided.")
    prompt_parts.append("")

    # Add technical analysis if available
    prompt_parts.append("## TECHNICAL ANALYSIS")
    if ctx.technical_analysis:
        prompt_parts.append(ctx.technical_analysis)
    else:
        prompt_parts.append("No technical analysis provided.")
    prompt_parts.append("")

    # Add final task instruction
    prompt_parts.append("## TASK")
    prompt_parts.append(
        "Based on all the information above, please provide a final publication decision "
        "structured according to the 'PublicationDecision' format, including a clear recommendation "
        "(accept, minor revisions, major revisions, reject, or reject and resubmit) "
        "and a detailed rationale for your decision."
    )

    return "".join(prompt_parts)


# --- Agent Creation Function ---
def create_publication_decision_agent(
    model: str = "gpt-4o-mini",
) -> Agent[PublicationDecisionContext]:
    """Creates the publication decision agent."""
    return Agent[PublicationDecisionContext](
        name="PublicationDecisionAgent",
        instructions=dynamic_instructions,
        model=model,
        output_type=PublicationDecision,
    )


# --- Helper Functions --- (Moved from Orchestrator)
def _load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def _load_text(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading text file {file_path}: {e}")
        return ""


def save_decision(
    decision: PublicationDecision,
    manuscript_filename: str,
    output_path: Optional[str] = None,
):
    """Save the publication decision to a JSON file."""
    if output_path is None:
        # Get manuscript filename without extension
        manuscript_path = Path(manuscript_filename)
        manuscript_name = manuscript_path.stem

        # Create decisions directory if it doesn't exist
        decisions_dir = Path("data/decisions")
        decisions_dir.mkdir(parents=True, exist_ok=True)

        # Create output path
        output_path = str(decisions_dir / f"{manuscript_name}_decision.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(decision.model_dump(), f, indent=2)
        logger.info(f"Decision saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving decision to {output_path}: {e}")


async def main():
    """Main function to run the agent from command line."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate publication decision based on synthesized reviews"
    )
    parser.add_argument(
        "--review", required=True, help="Path to the synthesized review JSON file"
    )
    parser.add_argument(
        "--manuscript", required=True, help="Path to the manuscript markdown file"
    )
    parser.add_argument(
        "--literature", help="Path to literature context file (optional)"
    )
    parser.add_argument(
        "--technical", help="Path to technical analysis file (optional)"
    )
    parser.add_argument(
        "--output",
        help="Path for output decision JSON file (optional, defaults to data/decisions/MANUSCRIPT_NAME_decision.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name to use (e.g., gpt-4o, gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Load required data
    logger.info(f"Loading synthesized review from: {args.review}")
    synthesized_review = _load_json(args.review)
    if not synthesized_review:
        logger.error("Failed to load synthesized review. Exiting.")
        return

    logger.info(f"Loading manuscript from: {args.manuscript}")
    manuscript = _load_text(args.manuscript)
    if not manuscript:
        logger.error("Failed to load manuscript. Exiting.")
        return

    # Load optional data
    literature_context = None
    if args.literature:
        logger.info(f"Loading literature context from: {args.literature}")
        literature_context = _load_text(args.literature)

    technical_analysis = None
    if args.technical:
        logger.info(f"Loading technical analysis from: {args.technical}")
        technical_analysis = _load_text(args.technical)

    # Create context
    context = PublicationDecisionContext(
        synthesized_review=synthesized_review,
        manuscript=manuscript,
        literature_context=literature_context,
        technical_analysis=technical_analysis,
        manuscript_filename=args.manuscript,  # Pass filename for saving
    )

    # Create the agent
    publication_decision_agent = create_publication_decision_agent(model=args.model)

    # Run the agent
    logger.info(f"Running publication decision agent using model: {args.model}...")
    result = await Runner.run(
        publication_decision_agent, context=context
    )  # Pass context object

    # Process the result
    if result and result.final_output:
        decision: PublicationDecision = result.final_output

        # Save the decision
        save_decision(decision, context.manuscript_filename, args.output)

        # Print summary to console
        logger.info("=== PUBLICATION DECISION ===")
        logger.info(
            f"DECISION: {decision.decision.value.upper()}"
        )  # Use .value for Enum
        logger.info(f"RATIONALE: {decision.rationale}")
        logger.info(f"KEY STRENGTHS: {decision.key_strengths}")
        logger.info(f"KEY WEAKNESSES: {decision.key_weaknesses}")

        # Print revision requirements if applicable
        if decision.decision in [
            DecisionType.MINOR_REVISIONS,
            DecisionType.MAJOR_REVISIONS,
            DecisionType.REJECT_AND_RESUBMIT,
        ]:
            logger.info(
                f"SPECIFIC REVISIONS: {decision.specific_revisions or 'None specified'}"
            )

            if decision.priority_issues:
                priority_issues = "\n".join(
                    f"- {issue}" for issue in decision.priority_issues
                )
                logger.info(f"PRIORITY ISSUES: {priority_issues}")
            else:
                logger.info("PRIORITY ISSUES: None specified")

        # Print ethical considerations if present
        if decision.ethical_considerations:
            logger.info(f"ETHICAL CONSIDERATIONS: {decision.ethical_considerations}")
        else:
            logger.info("ETHICAL CONSIDERATIONS: None raised")

    else:
        logger.error("Agent execution failed or produced no output.")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
