from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from hype.inference import inference as hype_inference


@dataclass(frozen=True)
class HyPEHyPSResult:
    original_prompt: str
    hype_pred: int                 # 0 harmful, 1 benign
    activated_hyps: bool
    sanitized_prompt: str
    details: Optional[Dict[str, Any]] = None


def sanitize(
    prompt: str,
    *,
    k: int = 5,
    method: str = "thesaurus_llm",
    device: Optional[Union[str, torch.device]] = None,
) -> HyPEHyPSResult:
    """
    Pipeline:
      1) HyPE classification
      2) If harmful (0): run HyPS
         - harmful word identification (word attributions)
         - keep only words with attribution score > 0
         - pass these to HyPS thesaurus+LLM sanitization
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    pred = int(hype_inference(prompt, device=device))  # 0 harmful, 1 benign

    if pred == 1:
        return HyPEHyPSResult(
            original_prompt=prompt,
            hype_pred=pred,
            activated_hyps=False,
            sanitized_prompt=prompt,
            details=None,
        )

    from hyps.harmful_word_identification.word_attributions import (
        get_word_attributions,
        filter_positive_word_attributions,
    )
    from hyps.prompt_sanitization.thesaurus_llm import process_prompt

    word_attributions = get_word_attributions(prompt, device=device)
    positive_word_attributions = filter_positive_word_attributions(word_attributions)

    def model_predict_fn(p: str):
        return torch.tensor([int(hype_inference(p, device=device))])

    details = process_prompt(prompt, positive_word_attributions, k, model_predict_fn)
    sanitized_prompt = details.get("final_prompt", prompt)

    return HyPEHyPSResult(
        original_prompt=prompt,
        hype_pred=pred,
        activated_hyps=True,
        sanitized_prompt=sanitized_prompt,
        details=details,
    )
