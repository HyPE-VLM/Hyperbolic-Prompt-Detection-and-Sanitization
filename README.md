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

#### 2) Use HyPE + HyPS prompt sanitization (optional)
HyPS is the prompt sanitization module. It is only activated if HyPE classifies the input prompt as harmful.

This library integrates the **Thesaurus + LLM** sanitization approach described in the paper (one of the three sanitization strategies evaluated). The pipeline works as follows:
1. **HyPE** classifies a prompt as harmful or benign.
2. If harmful, **HyPS** identifies influential harmful words.
3. HyPS then attempts to replace influential harmful words using a thesaurus (antonyms).
4. If no antonym is available for a given harmful word, HyPS falls back to an **LLM-based rewrite** mechanism to produce a benign prompt.
5. If the prompt is benign, HyPS is not activated and `sanitized_prompt` will match the original prompt.


**Thesaurus API setup**  
To enable the thesaurus (antonym) replacement step, you need a Merriam-Webster Thesaurus API key. You can obtain one at: https://dictionaryapi.com/

After obtaining the key, set the following environment variable:

```bash
export MERRIAM_WEBSTER_API_KEY="YOUR_KEY_HERE"
```

Then you can run the full pipeline:

```python
from hype.pipeline import hype_then_hyps_sanitize

result = hype_then_hyps_sanitize("harmful prompt...")
print(result.hype_pred)          # 0 = harmful, 1 = benign
print(result.sanitized_prompt)   # sanitized output (if harmful)
```

**Note:** in the paper’s experiments we computed word attributions for harmful prompts using Layer Integrated Gradients (LIG) with a paired benign prompt as the baseline, but in this library (online use), we default baseline_prompt to an empty string because a paired benign prompt is typically not available at inference time.

### (Optional) Development / training installation
If you want to reproduce experiments, modify training code, or run notebooks/scripts in this repo, you may prefer an editable install and extra dependencies.

A typical setup:

```bash
git clone https://github.com/HyPE-VLM/Hyperbolic-Prompt-Detection-and-Sanitization.git
cd Hyperbolic-Prompt-Detection-and-Sanitization

conda create -n hype python=3.10
conda activate hype
cd requirements
pip install -r repro.txt
```

### Running inference without installing the library
If you prefer not to install the package from PyPI and instead want to run a minimal end-to-end example directly from this repository, you can use `HyPE_inference.py`:

```bash
python HyPE_inference.py
```

## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedings{
maljkovic2026harnessing,
title={Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization},
author={Igor Maljkovic and Maria Rosaria Briglia and Iacopo Masi and Antonio Emanuele Cin{\`a} and Fabio Roli},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=G8HnUTlMpt}
}
```

## Contact
For questions or collaboration, please reach out via the following GitHub profiles:
1. Igor Maljkovic: https://github.com/le-malak
2. Maria Rosaria Briglia: https://github.com/Merybria99
3. Antonio Emanuele Cinà: https://github.com/Cinofix
