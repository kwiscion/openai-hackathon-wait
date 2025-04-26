import os

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def convert(path: str):
    """
    Convert a PDF file to markdown text and extract images.
    
    This function takes a PDF file path, extracts text content and images,
    creates a directory named after the PDF file, and saves the markdown text
    and images to that directory.
    
    Args:
        path (str): Path to the PDF file to convert.
        
    Returns:
        None: The function saves the extracted content to disk but doesn't return any value.
        
    Note:
        The markdown file will be saved as '{dirname}/{dirname}.md' where dirname is derived
        from the input PDF filename. Images will be saved in the same directory.
    """
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter("2405.17640v2.pdf")
    text, _, images = text_from_rendered(rendered)


    dirname = path.split("/")[-1].split(".")[0]
    os.makedirs(dirname, exist_ok=True)
    for i, image in enumerate(images):
        images[image].save(f"{dirname}/{image}")


    with open(f"{dirname}/{dirname}.md", "w", encoding='utf-8') as f:
        f.write(text)