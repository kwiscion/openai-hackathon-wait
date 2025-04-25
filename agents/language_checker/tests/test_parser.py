from agents.language_checker.parser import MarkdownStructureValidator
from agents.language_checker.types import SectionIssue, LanguageFeedback

def test_structure_validator():
    markdown_text = """
# Abstract
This is the abstract.

# Introduction
This is the introduction.

# Methods
Description of methods.

# Results
Some results here.

# Discussion
Discussion points.

# Conclusion
Final thoughts.
"""

    validator = MarkdownStructureValidator(content=markdown_text)
    result = validator.validate_structure()

    assert result.is_valid == True
    assert len(result.missing_sections) == 0
    assert "Abstract" in result.detected_sections
    print("Structure validation passed.")

def test_types():
    issues = [
        SectionIssue(section_name="Abstract", problems=["Grammar issue in first sentence."]),
        SectionIssue(section_name="Introduction", problems=["Long sentences."])
    ]
    feedback = LanguageFeedback(
        overall_opinion="Moderate language issues found.",
        issues=issues
    )
    assert feedback.overall_opinion == "Moderate language issues found."
    assert len(feedback.issues) == 2
    print("Types instantiation passed.")

if __name__ == "__main__":
    test_structure_validator()
    test_types()
