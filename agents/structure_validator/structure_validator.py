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
            str: A summary of the paper's structure and improvement recommendations.
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
                    "content": "You are a scientific paper structure validator. Analyze the given paper and identify which of the expected sections are present and which are missing. Provide only a brief summary and specific improvement recommendations without detailed analysis."
                },
                {
                    "role": "user",
                    "content": f"Analyze the structure of this scientific paper and identify which of these expected sections are present: {', '.join(expected_sections)}. Then provide ONLY: 1) A brief summary of present/missing sections, 2) Specific improvement recommendations. Be concise.\n\nPaper content:\n{paper_content}"
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        validation_result = response.choices[0].message.content
        return validation_result
    
    @function_tool
    def check_grammar_punctuation(paper_path: str) -> str:
        """
        Checks grammar and punctuation in a scientific paper.
        
        Args:
            paper_path (str): The file path to the scientific paper to check.
            
        Returns:
            str: A summary of grammar issues and improvement recommendations.
        """
        # Read paper content from file
        try:
            with open(paper_path, 'r', encoding='utf-8') as file:
                paper_content = file.read()
        except Exception as e:
            return f"Error reading paper file: {str(e)}"
        
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
                    "content": f"Review this scientific paper for grammar and punctuation issues. Provide ONLY: 1) A concise summary of overall writing quality, 2) 3-5 specific improvement recommendations with brief examples. Do not provide a detailed section-by-section analysis.\n\nPaper content:\n{paper_content}"
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        grammar_analysis = response.choices[0].message.content
        return grammar_analysis
    
    @function_tool
    def generate_final_summary(structure_analysis: str, grammar_analysis: str) -> str:
        """
        Generates a final concise summary combining structure and grammar analyses with a decision.
        
        Args:
            structure_analysis (str): The structure analysis of the paper.
            grammar_analysis (str): The grammar and punctuation analysis of the paper.
            
        Returns:
            str: A concise final summary with improvement recommendations and review decision.
        """
        # Use OpenAI to create a concise combined summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific paper reviewer who provides concise and actionable feedback. Create a very brief summary that combines structure and grammar analyses into a concise, organized final report, and make a clear decision about whether the paper should proceed to in-depth review."
                },
                {
                    "role": "user",
                    "content": f"Create a concise final summary that combines these two analyses. Include: 1) 'Overall Assessment' section with brief findings, 2) 'Key Recommendations' section with 3-5 actionable improvements, and 3) 'Review Decision' section with a clear YES or NO decision on whether the paper should proceed to in-depth review by another agent. The decision should be based on the severity of issues found - only papers with minor fixable issues should proceed.\n\nStructure Analysis:\n{structure_analysis}\n\nGrammar Analysis:\n{grammar_analysis}"
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        final_summary = response.choices[0].message.content
        return final_summary
    
    structure_validator_agent = Agent(
        name="Structure Validator",
        instructions="You are a structure validator for scientific papers. You will be given a paper and you will need to validate the structure of the paper and check for grammar/punctuation issues. Provide only concise summaries and specific recommendations without detailed analysis. Use the generate_final_summary tool to create a concise final report combining both analyses and include a clear decision about whether the paper should proceed to in-depth review by another agent.",
        model="gpt-4o-mini",
        tools=[validate_structure, check_grammar_punctuation, generate_final_summary],
    )
    
    result = await Runner.run(structure_validator_agent, input=f"Please analyze the paper at this path: {PAPER_PATH}. First check its structure, then check grammar and punctuation, and finally provide a concise summary report with the most important findings, recommendations, and a clear decision whether the paper should proceed to in-depth review by another agent.")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())