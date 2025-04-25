import asyncio
from agents.structure_validator.structure_validator import run_validator

async def test_validator():
    # Example markdown content for testing
    test_markdown = """# Neural Networks in Medical Imaging

## Abstract
This paper explores the application of neural networks in medical imaging analysis. We examine various architectures and their effectiveness in diagnosis support systems.

## Introduction
Medical imaging plays a crucial role in modern healthcare. The integration of artificial intelligence, particularly neural networks, has shown promising results in enhancing diagnostic accuracy.

## Literature Review
Previous studies have demonstrated the potential of convolutional neural networks in identifying patterns in radiological images. Smith et al. (2018) achieved 92% accuracy in pneumonia detection.

## Methodology
We implemented three neural network architectures: CNN, U-Net, and ResNet. Each model was trained on a dataset of 10,000 labeled X-ray images using cross-validation.

## Results
The U-Net architecture achieved the highest accuracy (94.3%) and sensitivity (92.1%) in identifying abnormalities. Table 1 presents a comparative analysis of all models.

## Discussion
Our findings suggest that segmentation-based approaches outperform traditional classification methods for medical image analysis. This may be attributed to their ability to preserve spatial information.

## Conclusion
Neural networks demonstrate significant potential for enhancing medical imaging analysis. Future work should focus on interpretability and reducing computational requirements.

## References
1. Smith, J., et al. (2018). Deep learning for automated pneumonia detection.
2. Johnson, R. (2020). U-Net architecture for medical image segmentation.
"""

    print("Testing structure validator with direct markdown input...")
    result = await run_validator(
        paper_content=test_markdown,
        auto_detect=True,
        grammar_check=True,
        min_score=5
    )
    
    print("\nValidator Output:")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_validator()) 