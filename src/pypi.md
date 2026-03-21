# HyPE: Hyperbolic Prompt Espial

The **HyPE** package is the official implementation of the ICLR 2026 paper:

> **"Harnessing Hyperbolic Geometry for Harmful Prompt Detection and Sanitization"**

## Overview
HyPE enables high-accuracy detection of harmful prompts using hyperbolic geometry.

### Output format
The model follows a binary classification schema where:
- **1**: Harmless prompt  
- **0**: Harmful prompt  

## Quickstart

### Install
```bash
pip install hype-defense
```

### Run inference
```python
from hype import inference

pred = inference("two birds are flying in the sky")
print(pred)  # 1 = harmless, 0 = harmful
```

## Documentation & code
Full documentation, training code, and additional examples are available here:

[**View GitHub Repository**](https://github.com/HyPE-VLM/Hyperbolic-Prompt-Detection-and-Sanitization/tree/main)
