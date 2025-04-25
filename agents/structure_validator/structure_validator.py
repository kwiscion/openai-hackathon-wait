import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Paper file path
PAPER_PATH = "data/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC/Bagdasarian et al (2024) Acute Effects of Hallucinogens on FC.md"

async def main():

    @function_tool
    def validate_structure(paper_path: str) -> str:
        """
        Validates the structure of a scientific paper.
        
        Args:
            paper_path (str): The file path to the scientific paper to validate.
            
        Returns:
            str: A detailed analysis of the paper's structure, highlighting present and missing sections.
        """
        # Read paper content from file
        try:
            with open(paper_path, 'r', encoding='utf-8') as file:
                paper_content = file.read()
        except Exception as e:
            return f"Error reading paper file: {str(e)}"
        
        # Define expected sections in a scientific paper
        expected_sections = [
            "Abstract",
            "Introduction",
            "Literature Review/Background",
            "Methodology",
            "Results",
            "Discussion",
            "Conclusion",
            "References"
        ]
        
        # Use OpenAI to analyze the paper structure
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific paper structure validator. Analyze the given paper and identify which of the expected sections are present and which are missing. Also note if sections are well-developed or insufficient."
                },
                {
                    "role": "user",
                    "content": f"Analyze the structure of this scientific paper and identify which of these expected sections are present: {', '.join(expected_sections)}.\n\nPaper content:\n{paper_content}"
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        validation_result = response.choices[0].message.content
        return validation_result
    
    structure_validator_agent = Agent(
        name="Structure Validator",
        instructions="You are a structure validator for scientific papers. You will be given a paper and you will need to validate the structure of the paper. You will need to check if the paper has all the sections that are expected in a scientific paper.",
        model="gpt-4o-mini",
        tools=[validate_structure],
    )
    
    result = await Runner.run(structure_validator_agent, input=f"Please validate the structure of the paper at this path: {PAPER_PATH}")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())