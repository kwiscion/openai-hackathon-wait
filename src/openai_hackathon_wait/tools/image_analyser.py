from openai_hackathon_wait.agents.image_analysis.utils import encode_image
from openai_hackathon_wait.agents.image_analysis.image_analyser import (
    FigureExtractorContext,
    IMAGE_DESCRIPTION_PROMPT,
    client,
)
from agents import function_tool


@function_tool
def image_analysis(figure_extractor_context: FigureExtractorContext) -> str:
    encoded_image = encode_image(figure_extractor_context["image_paths"])
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": IMAGE_DESCRIPTION_PROMPT},
                    {
                        "type": "input_text",
                        "text": figure_extractor_context["additional_context"],
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                ],
            }
        ],
    )
    return response.output_text
