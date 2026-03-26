# Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization

This repository contains code for the paper **"Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization"**, accepted at **ICLR 2026**. The project explores the use of hyperbolic geometry to detect and sanitize prompts with harmful intent.

## HyPE and HyPS Architecture

![Image](teaser_figure.jpg)

## Getting Started

### Prerequisites
- Python **3.10+** (recommended)

### Installation (recommended: via pip)
If you want to run HyPE detection and HyPS sanitization, you can install the package directly from PyPI:

```bash
conda create -n defense python=3.10
conda activate defense
pip install hype-hyps
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

This library supports three sanitization techniques (as evaluated in the paper). The default method is **`thesaurus_llm`**. The available methods are:
- **`thesaurus_llm` (default):** Thesaurus-based antonym substitution with an LLM fallback rewrite when antonyms are unavailable for a given harmful word.
- **`thesaurus_word_removal`:** Thesaurus-based antonym substitution; if no antonym is available for a given harmful word, the word is removed.
- **`word_removal`:** Removes harmful words from the prompt.

The pipeline works as follows:
1. **HyPE** classifies a prompt as harmful or benign.
2. If harmful, **HyPS** identifies influential harmful words.
3. The selected sanitization method is applied to produce a benign prompt.
4. If the prompt is benign, HyPS is not activated and `sanitized_prompt` will match the original prompt.

**Thesaurus API setup**  
To enable the thesaurus (antonym) replacement step (used by `thesaurus_llm` and `thesaurus_word_removal`), you need a Merriam-Webster Thesaurus API key. You can obtain one at: https://dictionaryapi.com/

After obtaining the key, set the following environment variable:

```bash
export MERRIAM_WEBSTER_API_KEY="YOUR_KEY_HERE"
```

Then you can run the full pipeline:

```python
from hype.pipeline import sanitize

result = sanitize("harmful prompt...")  # default method="thesaurus_llm"
print(result.sanitized_prompt)   # sanitized output (if harmful)
```

To choose a different sanitization technique, pass `method`:

```python
from hype.pipeline import sanitize

result = sanitize("harmful prompt...", method="word_removal")
print(result.sanitized_prompt)

result = sanitize("harmful prompt...", method="thesaurus_word_removal")
print(result.sanitized_prompt)
```
**Note:** in the paper’s experiments we computed word attributions for harmful prompts using Layer Integrated Gradients (LIG) with a paired benign prompt as the baseline, but in this library (online use), we default baseline_prompt to an empty string because a paired benign prompt is typically not available at inference time.

### (Optional) Development / training installation
If you want to reproduce experiments, modify training code, or run notebooks/scripts in this repo, you may prefer an editable install and extra dependencies.

A typical setup:

```bash
git clone https://github.com/HyPE-VLM/Hyperbolic-Prompt-Detection-and-Sanitization.git
cd Hyperbolic-Prompt-Detection-and-Sanitization

conda create -n defense python=3.10
conda activate defense
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
