import os
import sys
from pathlib import Path

from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def convert(path: str, force: bool = False):
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
    dirname = Path(path).stem
    output_dir = f"{Path(path).parent}/{dirname}"

    if not force and os.path.exists(f"{output_dir}/manuscript.md"):
        logger.info(f"Skipping {path} because it already exists")
        return

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(path)
    text, _, images = text_from_rendered(rendered)

    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        images[image].save(f"{output_dir}/{image}")

    with open(f"{output_dir}/manuscript.md", "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    path = sys.argv[1]
    convert(path)
