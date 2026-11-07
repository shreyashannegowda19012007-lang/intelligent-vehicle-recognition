"""
Model Checkpoint Reassembly and Management Utilities.
Handles automatic reconstruction of chunked model checkpoints for git compliance.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_convnext_checkpoint(root: Path | str | None = None) -> Path:
    """
    Ensure models/classification/convnext_best.pt exists.
    If absent, reassemble from git-tracked chunks (.part0, .part1).

    Returns:
        Path to resolved convnext_best.pt checkpoint.
    """
    if root is None:
        try:
            from src.utils.config import find_project_root
            root_path = find_project_root()
        except Exception:
            root_path = Path(__file__).resolve().parent.parent.parent
    else:
        root_path = Path(root).resolve()

    target = root_path / "models" / "classification" / "convnext_best.pt"
    if target.is_file() and target.stat().st_size > 50 * 1024 * 1024:
        return target

    p0 = root_path / "models" / "classification" / "convnext_best.pt.part0"
    p1 = root_path / "models" / "classification" / "convnext_best.pt.part1"

    if p0.is_file() and p1.is_file():
        logger.info("Reassembling ConvNeXt-Tiny weights from split chunks (%s, %s)...", p0.name, p1.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f_out:
            with open(p0, "rb") as f0:
                f_out.write(f0.read())
            with open(p1, "rb") as f1:
                f_out.write(f1.read())
        logger.info("Successfully reassembled %s (%.2f MB).", target.name, target.stat().st_size / (1024 * 1024))

    return target
