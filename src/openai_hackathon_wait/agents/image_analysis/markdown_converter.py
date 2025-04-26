import os
from pathlib import Path

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
    rendered = converter(path)
    text, _, images = text_from_rendered(rendered)

    dirname = Path(path).stem
    os.makedirs(f"{Path(path).parent}/{dirname}", exist_ok=True)
    for i, image in enumerate(images):
        images[image].save(f"{Path(path).parent}/{dirname}/{image}")


    with open(f"{Path(path).parent}/{dirname}/{dirname}.md", "w", encoding='utf-8') as f:
        f.write(text)