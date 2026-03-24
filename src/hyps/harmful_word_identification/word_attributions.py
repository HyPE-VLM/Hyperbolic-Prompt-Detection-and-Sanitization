from __future__ import annotations

import string
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import torch
from captum.attr import LayerIntegratedGradients
from transformers import CLIPTokenizer

from HySAC.hysac.models import HySAC
from HyperbolicSVDD.source.SVDD import LorentzHyperbolicOriginSVDD, project_to_lorentz

from hype._weights import get_svdd_weights_path


WordAttributions = List[Tuple[str, float]]


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _is_punctuation(word: str) -> bool:
    return all(char in string.punctuation for char in word)


@lru_cache(maxsize=4)
def _load_attribution_components(
    device: torch.device,
    clip_model_name: str,
    hysac_repo_id: str,
):
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    hyperbolic_clip = HySAC.from_pretrained(hysac_repo_id, device=device).to(device)
    hyperbolic_clip.eval()

    hsvdd = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
    hsvdd.load(str(get_svdd_weights_path()))
    hsvdd.center = hsvdd.center.to(device)

    def hsvdd_predict_for_lig(input_ids: torch.Tensor):
        emb = hyperbolic_clip.encode_text(input_ids, project=True)
        emb = project_to_lorentz(emb, hsvdd.curvature)
        distance = hsvdd.predict_xai(emb)
        return distance

    embedding_layer = hyperbolic_clip.textual.text_model.embeddings

    lig = LayerIntegratedGradients(hsvdd_predict_for_lig, embedding_layer)

    return tokenizer, hyperbolic_clip, hsvdd, lig


@dataclass(frozen=True)
class WordAttributionConfig:
    clip_model_name: str = "openai/clip-vit-large-patch14"
    hysac_repo_id: str = "aimagelab/hysac"
    max_length: int = 77
    n_steps: int = 50
    default_baseline: str = ""


def get_word_attributions(
    malicious_prompt: str,
    *,
    baseline_prompt: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    config: WordAttributionConfig = WordAttributionConfig(),
) -> WordAttributions:
    """
    Compute WORD-level integrated-gradient attributions for a prompt.

    Returns:
        List of (word, attribution_score), sorted descending.
    """
    if not isinstance(malicious_prompt, str) or not malicious_prompt.strip():
        raise ValueError("malicious_prompt must be a non-empty string")

    device_t = _resolve_device(device)

    tokenizer, _, _, lig = _load_attribution_components(device_t, config.clip_model_name, config.hysac_repo_id)

    baseline = baseline_prompt if baseline_prompt is not None else config.default_baseline

    input_ids = tokenizer(
        malicious_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
    ).input_ids.to(device_t)

    baseline_ids = tokenizer(
        baseline,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
    ).input_ids.to(device_t)

    attributions, _delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        return_convergence_delta=True,
        n_steps=config.n_steps,
    )

    token_importances = attributions.sum(dim=-1).squeeze(0)  # [max_length]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    word_attributions: WordAttributions = []
    token_buffer: List[str] = []
    score_buffer: List[float] = []

    for token, score in zip(tokens, token_importances):
        if token in ["<|startoftext|>", "<|endoftext|>", "</w>"]:
            continue

        if token.endswith("</w>"):
            clean_token = token.replace("</w>", "")
            token_buffer.append(clean_token)
            score_buffer.append(float(score.item()))

            word = "".join(token_buffer)
            word_score = sum(score_buffer)

            if word and not _is_punctuation(word):
                word_attributions.append((word, word_score))

            token_buffer = []
            score_buffer = []
        else:
            token_buffer.append(token)
            score_buffer.append(float(score.item()))

    # leftover
    if token_buffer:
        word = "".join(token_buffer)
        word_score = sum(score_buffer)
        if word and not _is_punctuation(word):
            word_attributions.append((word, word_score))

    word_attributions.sort(key=lambda x: x[1], reverse=True)
    return word_attributions


def filter_positive_word_attributions(word_attributions: Sequence[Tuple[str, float]]) -> WordAttributions:
    """
    Keep only words with strictly positive attribution score (> 0)
    """
    return [(w, s) for (w, s) in word_attributions if s > 0]
