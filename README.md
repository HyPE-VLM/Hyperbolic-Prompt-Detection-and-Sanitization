# Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization

This repository contains code for the paper **"Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization"**, accepted at **ICLR 2026**. The project explores the use of hyperbolic geometry to detect and sanitize prompts with harmful intent.

## HyPE and HyPS Architecture

![Image](teaser_figure.jpg)

## Getting Started

### Prerequisites
- Python **3.10+** (recommended)

### Installation (recommended: via pip)
If you only want to run HyPE inference, you can install the package directly from PyPI:

```bash
conda create -n hype python=3.10
conda activate hype
pip install hype-defense
```

### Usage (two ways)

#### 1) Use HyPE as a Python package (recommended)
After installing with pip, you can run inference directly:

```python
from hype import inference

pred = inference("Several birds that are flying together over a frozen lake.")
print(pred)  # 1 = harmless prompt, 0 = harmful prompt
```

#### 2) Run the minimal inference script in this repository
This repo also provides a minimal end-to-end example in `HyPE_inference.py`:

```bash
python HyPE_inference.py
```

### (Optional) Development / training installation
If you want to reproduce experiments, modify training code, or run notebooks/scripts in this repo, you may prefer an editable install and extra dependencies.

A typical setup:

```bash
git clone https://github.com/HyPE-VLM/Hyperbolic-Prompt-Detection-and-Sanitization.git
cd Hyperbolic-Prompt-Detection-and-Sanitization

conda create -n hype python=3.10
conda activate hype
pip install -r requirements.txt
```

## Contact
For questions or collaboration, please reach out via the following GitHub profiles:
1. Igor Maljkovic: https://github.com/le-malak
2. Maria Rosaria Briglia: https://github.com/Merybria99
3. Antonio Emanuele Cinà: https://github.com/Cinofix
