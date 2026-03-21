from __future__ import annotations

from importlib import resources
from pathlib import Path


def get_svdd_weights_path() -> Path:
    with resources.as_file(resources.files("hype") / "assets" / "best_hyperbolic_svdd_model.pth") as p:
        return Path(p)
