from typing import List, Optional
import asyncio
import io
import base64

import pymupdf4llm
from loguru import logger
from pdf2image import convert_from_path, convert_from_bytes
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class ExtractedText(BaseModel):
    """Text extracted from an image."""
    text: str = Field(description="Text extracted from the image")


async def extract_text_from_image(image, model, semaphore):
    """
    Extract text from a single image using OpenAI vision model.
    
    Args:
        image: The image to extract text from
        model: The LangChain model to use
        semaphore: Asyncio semaphore for concurrency control
        
    Returns:
        Extracted text from the image
    """
    async with semaphore:
        # Convert image to bytes and encode as base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        
        structured_llm = model.with_structured_output(ExtractedText)
        
        messages = [
            SystemMessage(content="""You are an expert at extracting text from images.
                 Your task is to extract all visible text from the provided image accurately.
                 Maintain proper paragraph structure and formatting where possible."""),
            HumanMessage(content=[
                {"type": "text", "text": "Extract all text from this image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
        ]
        
        try:
            response = await structured_llm.ainvoke(messages)
            logger.info(f"Successfully extracted text from image")
            return response.text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""


async def get_text_from_pdf_with_vision(pdf_path: str = None, pdf_bytes: bytes = None, max_concurrent: int = 3) -> str:
    """
    Extract text from a PDF by converting to images and using OpenAI vision model.
    
    Args:
        pdf_path: Path to the PDF file (optional)
        pdf_bytes: PDF file as bytes (optional)
        max_concurrent: Maximum number of concurrent page processing
        
    Returns:
        Extracted text from the PDF
    """
    if not pdf_path and not pdf_bytes:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")
    
    # Convert PDF to images
    images = []
    if pdf_path:
        images = convert_from_path(pdf_path)
    else:
        images = convert_from_bytes(pdf_bytes)
    
    logger.info(f"PDF converted to {len(images)} images")
    
    model = ChatOpenAI(model="gpt-4o", temperature=0, timeout=60)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process images concurrently
    tasks = [extract_text_from_image(image, model, semaphore) for image in images]
    results = await asyncio.gather(*tasks)
    
    # Combine text from all pages
    full_text = "\n\n".join([text for text in results if text])
    
    return full_text


async def get_text_from_pdf_with_pymupdf(pdf_path: str = None) -> str:
    """
    Extract text from a PDF using pymupdf4llm.

    Args:
        pdf_path: Path to the PDF file (optional)
        pdf_bytes: PDF file as bytes (optional)
        max_concurrent: Maximum number of concurrent page processing
        
    Returns:
        Extracted text from the PDF
    """
    if not pdf_path:
        raise ValueError("pdf_path must be provided")
    
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text
