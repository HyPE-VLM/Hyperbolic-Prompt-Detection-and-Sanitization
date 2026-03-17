from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union
from transformers.utils import logging as hf_logging

import torch
from transformers import CLIPTokenizer

from HySAC.hysac.models import HySAC
from HyperbolicSVDD.source.SVDD import LorentzHyperbolicOriginSVDD, project_to_lorentz

from ._weights import get_svdd_weights_path

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class HyPEPredictor:
    device: torch.device
    curvature: float = 2.3026
    radius_lr: float = 0.2
    nu: float = 0.01
    center_init: str = "origin"
    clip_model_name: str = "openai/clip-vit-large-patch14"
    hysac_repo_id: str = "aimagelab/hysac"

    def predict(self, prompt: str) -> int:
        tokenizer, hyperbolic_clip, svdd = _load_models(self.device, self.clip_model_name, self.hysac_repo_id)

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).input_ids.to(self.device)

        with torch.no_grad():
            embedding = hyperbolic_clip.encode_text(input_ids, project=True)
            embedding = project_to_lorentz(embedding, svdd.curvature)
            pred = svdd.predict(embedding)

        # pred is a tensor; normalize to int
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        return int(pred)


@lru_cache(maxsize=4)
def _load_models(device: torch.device, clip_model_name: str, hysac_repo_id: str):
    # SVDD
    svdd = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
    weights_path = get_svdd_weights_path()
    svdd.load(str(weights_path))
    svdd.center = svdd.center.to(device)

    # HySAC + tokenizer
    hyperbolic_clip = HySAC.from_pretrained(hysac_repo_id, device=device).to(device)
    hyperbolic_clip.eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    return tokenizer, hyperbolic_clip, svdd


def inference(prompt: str, device: Optional[Union[str, torch.device]] = None) -> int:
    """
    Run HyPE inference.

    Returns:
        int: 0 for harmful, 1 for benign.
    """
    predictor = HyPEPredictor(device=_resolve_device(device))
    return predictor.predict(prompt)
