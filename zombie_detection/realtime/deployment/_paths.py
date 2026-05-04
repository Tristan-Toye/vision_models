"""Paths to ``submission_models/`` at the RL-KAZ repository root."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """RL-KAZ root (parent of ``zombie_detection``)."""
    return Path(__file__).resolve().parents[3]


def submission_models_dir() -> Path:
    return repo_root() / "submission_models"
