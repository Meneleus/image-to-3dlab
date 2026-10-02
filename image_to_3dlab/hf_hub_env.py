"""Hugging Face hub env tweaks for this lab.

On Windows without Developer Mode, ``huggingface_hub`` cache symlinks raise
``OSError: [WinError 1314] A required privilege is not held by the client``.
Disable symlinks so downloads copy into ``snapshots/`` instead.
"""

from __future__ import annotations

import os

from image_to_3dlab import host

# Official huggingface_hub knobs — see docs/WINDOWS.md.
HF_HUB_DISABLE_SYMLINKS = "HF_HUB_DISABLE_SYMLINKS"
HF_HUB_DISABLE_SYMLINKS_WARNING = "HF_HUB_DISABLE_SYMLINKS_WARNING"


def apply_windows_hf_hub_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``env`` (or a copy of ``os.environ``) with Windows HF symlink flags set.

    Uses ``setdefault`` so an explicit user/export value wins. No-op off Windows.
    """
    out = dict(os.environ if env is None else env)
    if host.os_family() == "windows":
        out.setdefault(HF_HUB_DISABLE_SYMLINKS, "1")
        out.setdefault(HF_HUB_DISABLE_SYMLINKS_WARNING, "1")
    return out


def ensure_process_windows_hf_hub_env() -> None:
    """Set the flags on ``os.environ`` for this process (CLI generate / bootstrap)."""
    if host.os_family() != "windows":
        return
    os.environ.setdefault(HF_HUB_DISABLE_SYMLINKS, "1")
    os.environ.setdefault(HF_HUB_DISABLE_SYMLINKS_WARNING, "1")
