from typing import List
from dataclasses import dataclass


@dataclass
class SectionIssue:
    section_name: str
    problems: List[str]


@dataclass
class LanguageFeedback:
    overall_opinion: str
    issues: List[SectionIssue]
