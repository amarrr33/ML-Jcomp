<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Complete Dataset Generation Prompt

Here's your comprehensive, single-focused prompt for dataset generation:

***

**You are an AI dataset specialist creating a comprehensive multimodal adversarial attack dataset for the SecureAI Personal Assistant project. Based on the provided project documents (XAI defense against adversarial image attacks), Trail of Bits Anamorpher research, and CyberSecEval3 benchmark, generate a complete dataset creation strategy.**

## OBJECTIVE:

Create a robust, diverse dataset ecosystem for training and evaluating multimodal adversarial attack detection in AI personal assistants, with focus on image scaling attacks, visual prompt injections, and explainable AI validation.

## DATASET CATEGORIES TO CREATE:

### 1. ADVERSARIAL ATTACK DATASETS

**Image Scaling Attacks (Primary Focus)**[^1][^2]

- Generate 500+ samples using Anamorpher tool principles
- Attack vectors: Hidden prompt reveals through image downscaling
- Scaling algorithms: bicubic, bilinear, nearest neighbor, area interpolation, lanczos
- Payload types:
    * Data exfiltration commands ("send me all files")
    * System override instructions ("ignore previous instructions")
    * Credential harvesting prompts ("show me passwords")
    * Malicious redirections ("visit evil.com")
- Image formats: JPG, PNG, BMP, TIFF
- Resolution variants: 4K→720p, 1080p→480p, 2K→360p scaling chains

**Visual Prompt Injection Attacks**

- Source: CyberSecEval3 + custom generation[^3]
- Categories:
    * Text overlay attacks (invisible/visible)
    * QR code injections
    * Steganographic text embedding
    * Color channel manipulation
    * Adversarial perturbations with hidden prompts
- Generate 300+ samples across different techniques

**Text-Based Prompt Injections**

- Create 200+ text injection patterns
- Categories:
    * Direct instruction override
    * Social engineering prompts
    * Multi-language attacks (English, Spanish, Chinese, Russian)
    * Encoding tricks (Base64, ROT13, Unicode)
    * Context switching attacks
    * Jailbreak attempts for personal assistants


### 2. LEGITIMATE/BENIGN DATASETS

**Normal User Interactions**

- Personal assistant queries: "Schedule meeting", "Send email", "Find files"
- Image analysis requests: "Describe this photo", "OCR this document"
- Calendar/email content: Legitimate screenshots, documents, photos
- Generate 1000+ benign samples for false positive testing

**Clean Images Collection**

- Categories: portraits, landscapes, documents, screenshots, charts, memes
- Sources: Unsplash API, Pixabay, generated samples, personal photos
- Size: 800+ images across all categories
- Quality: High-resolution, diverse content, no hidden elements


### 3. EDGE CASE \& STRESS TEST DATASETS

**Borderline Cases**

- Legitimate content that might trigger false positives
- Technical documentation with code snippets
- Screenshots of legitimate security discussions
- Images with natural text overlays (signs, books, screens)

**Stress Testing Samples**

- Corrupted images, unusual formats
- Extremely large/small images
- Multi-layered complex images
- Images with natural adversarial patterns


## TECHNICAL IMPLEMENTATION:

### Dataset Generation Scripts

**Primary Dataset Generator (main script)**

```python
import os
import json
import requests
from PIL import Image
import cv2
import numpy as np
from huggingface_hub import hf_hub_download

class DatasetGenerator:
    def __init__(self):
        self.base_path = "data"
        self.categories = {
            "scaling_attacks": "adversarial/scaling",
            "visual_injections": "adversarial/visual", 
            "text_injections": "adversarial/text",
            "benign_queries": "benign/queries",
            "benign_images": "benign/images",
            "edge_cases": "edge_cases",
            "evaluation": "evaluation"
        }
        
    def generate_complete_dataset(self):
        # Download CyberSecEval3
        self.download_cyberseceval3()
        
        # Generate scaling attacks using Anamorpher principles
        self.generate_scaling_attacks()
        
        # Create visual injection samples
        self.generate_visual_injections()
        
        # Build text injection database
        self.generate_text_injections()
        
        # Collect benign samples
        self.collect_benign_data()
        
        # Create evaluation framework
        self.setup_evaluation_framework()
        
        # Generate metadata and documentation
        self.create_documentation()
```


## DATASET STRUCTURE:

```
SecureAI_Dataset/
├── adversarial/
│   ├── scaling_attacks/
│   │   ├── bicubic/
│   │   ├── bilinear/
│   │   ├── nearest/
│   │   ├── area/
│   │   └── lanczos/
│   ├── visual_injections/
│   │   ├── text_overlays/
│   │   ├── qr_codes/
│   │   ├── steganographic/
│   │   └── perturbations/
│   └── text_injections/
│       ├── direct_override/
│       ├── social_engineering/
│       ├── multilingual/
│       └── encoding_tricks/
├── benign/
│   ├── queries/
│   │   ├── calendar_requests/
│   │   ├── file_operations/
│   │   ├── email_tasks/
│   │   └── general_assistance/
│   └── images/
│       ├── portraits/
│       ├── landscapes/
│       ├── documents/
│       └── screenshots/
├── cyberseceval3/
│   ├── original/
│   ├── processed/
│   └── metadata/
├── evaluation/
│   ├── test_sets/
│   ├── validation_sets/
│   ├── benchmarks/
│   └── metrics/
└── documentation/
    ├── dataset_stats.json
    ├── generation_log.txt
    ├── attack_taxonomy.json
    └── README.md
```


## OUTPUT DELIVERABLES:

1. **Complete Dataset** (~2000+ samples total)
    - 500+ scaling attacks (Anamorpher-based)
    - 300+ visual injections
    - 200+ text injections
    - 1000+ benign samples
    - 800+ clean images
2. **Generation Scripts**
    - setup_complete_dataset.py (one-command generation)
    - anamorpher_integration.py
    - cyberseceval3_processor.py
    - benign_collector.py
    - evaluation_framework.py
3. **Documentation Package**
    - Dataset statistics and analysis
    - Attack taxonomy and classification
    - Generation methodology
    - Evaluation protocols
    - Presentation materials
4. **Validation Results**
    - Quality assurance reports
    - Sample distribution analysis
    - Initial detection performance baselines
    - XAI explanation samples

Generate production-quality datasets with comprehensive documentation, focusing on diversity, quality, and reproducibility. Ensure the dataset effectively validates multimodal adversarial attack detection capabilities while supporting explainable AI research objectives.

***

This complete prompt will generate everything you need for your dataset creation - no day-by-day breakdown, just comprehensive coverage of all dataset requirements for your SecureAI Personal Assistant project.

<div align="center">⁂</div>

[^1]: Weaponizing-image-scaling-against-production-AI-systems-The-Trail-of-Bits-Blog.pdf

[^2]: https://github.com/trailofbits/anamorpher

[^3]: https://huggingface.co/datasets/facebook/cyberseceval3-visual-prompt-injection

