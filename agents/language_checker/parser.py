from typing import List, Dict
from dataclasses import dataclass
import re


@dataclass
class StructureValidationResult:
    is_valid: bool
    missing_sections: List[str]
    detected_sections: List[str]
    sections_content: Dict[str, str]


class MarkdownStructureValidator:
    EXPECTED_SECTIONS = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Conclusion"
    ]

    def __init__(self, content: str):
        self.content = content
        self.sections_content = self.extract_sections()

    def extract_sections(self) -> Dict[str, str]:
        pattern = r'^(#{1,6})\s+(.*)$'
        matches = list(re.finditer(pattern, self.content, re.MULTILINE))
        sections = {}
        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(self.content)
            section_text = self.content[start:end].strip()
            sections[header] = section_text
        return sections

    def validate_structure(self) -> StructureValidationResult:
        found_sections = list(self.sections_content.keys())
        found_lower = [s.lower() for s in found_sections]
        missing_sections = [
            expected for expected in self.EXPECTED_SECTIONS
            if expected.lower() not in found_lower
        ]
        is_valid = len(missing_sections) == 0
        return StructureValidationResult(
            is_valid=is_valid,
            missing_sections=missing_sections,
            detected_sections=found_sections,
            sections_content=self.sections_content
        )

    def has_minimal_structure(self) -> bool:
        found_lower = [s.lower() for s in self.sections_content.keys()]
        return "abstract" in found_lower and "methods" in found_lower
