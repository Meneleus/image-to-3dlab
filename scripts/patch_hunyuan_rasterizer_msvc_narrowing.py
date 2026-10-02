#!/usr/bin/env python3
"""Fix MSVC C2398 narrowing in Hunyuan ``custom_rasterizer`` ``grid_neighbor.cpp``.

On Windows / VS 2022–18 with ``/std:c++20``, brace-init of libtorch shapes from
``size_t`` (``vector::size()``, ``size()/n``) into ``int64_t`` dims is a
narrowing conversion (C2398). Cast those sizes to ``int64_t``. Harmless on Linux.

Only exact anchors are replaced; re-running is idempotent. Patterns that appear
in both ``build_hierarchy`` and ``build_hierarchy_with_feat`` are replaced in
all occurrences.

    python scripts/patch_hunyuan_rasterizer_msvc_narrowing.py
    python scripts/patch_hunyuan_rasterizer_msvc_narrowing.py \\
        --root vendor/hunyuan3d-cuda/hy3dpaint/custom_rasterizer
    python scripts/patch_hunyuan_rasterizer_msvc_narrowing.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "hunyuan3d-cuda" / "hy3dpaint" / "custom_rasterizer"

GRID_NEIGHBOR = "lib/custom_rasterizer_kernel/grid_neighbor.cpp"

# Longer ``seq2pos.size() / 3, 3`` must come before the single-dim form so we
# do not partially rewrite inside the two-dim brace-init.
REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        GRID_NEIGHBOR,
        "torch::zeros({seq2pos.size() / 3, 3}, float_options)",
        "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3), static_cast<int64_t>(3)}, float_options)",
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({seq2pos.size() / 3}, float_options)",
        "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3)}, float_options)",
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options)",
        (
            "torch::zeros({static_cast<int64_t>(seq2feat.size() / feat_channel), "
            "static_cast<int64_t>(feat_channel)}, float_options)"
        ),
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({grids[i].seq2grid.size(), 9}, int64_options)",
        (
            "torch::zeros({static_cast<int64_t>(grids[i].seq2grid.size()), "
            "static_cast<int64_t>(9)}, int64_options)"
        ),
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({grids[i].seq2evencorner.size()}, int64_options)",
        "torch::zeros({static_cast<int64_t>(grids[i].seq2evencorner.size())}, int64_options)",
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({grids[i].seq2oddcorner.size()}, int64_options)",
        "torch::zeros({static_cast<int64_t>(grids[i].seq2oddcorner.size())}, int64_options)",
    ),
    (
        GRID_NEIGHBOR,
        "torch::zeros({grids[i].downsample_seq.size()}, int64_options)",
        "torch::zeros({static_cast<int64_t>(grids[i].downsample_seq.size())}, int64_options)",
    ),
)


def patch_source(source: str, old: str, new: str) -> tuple[str, bool]:
    """Apply one replacement to every occurrence. Idempotent if already patched."""
    if old not in source:
        if new in source:
            return source, False
        raise RuntimeError(f"anchor missing for Hunyuan rasterizer MSVC patch:\n  {old}")
    return source.replace(old, new), True


def apply_to_tree(root: Path) -> list[str]:
    """Patch known sites under a custom_rasterizer checkout. Returns status lines."""
    by_file: dict[str, list[tuple[str, str]]] = {}
    for rel, old, new in REPLACEMENTS:
        by_file.setdefault(rel, []).append((old, new))

    states: list[str] = []
    for rel, pairs in by_file.items():
        path = root / rel
        if not path.is_file():
            states.append(f"MISSING {path}")
            continue
        text = path.read_text(encoding="utf-8")
        changed_any = False
        for old, new in pairs:
            text, changed = patch_source(text, old, new)
            changed_any = changed_any or changed
        if changed_any:
            path.write_text(text, encoding="utf-8")
            states.append(f"PATCHED {path}")
        else:
            states.append(f"APPLIED {path}")
    return states


def check_tree(root: Path) -> list[str]:
    """Report whether each site is already patched. Does not write."""
    states: list[str] = []
    for rel, old, new in REPLACEMENTS:
        path = root / rel
        if not path.is_file():
            states.append(f"MISSING {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if new in text and old not in text:
            states.append(f"APPLIED {path} :: {new[:48]}…")
        elif old in text:
            states.append(f"ABSENT {path} :: {old[:48]}…")
        else:
            states.append(f"UNKNOWN {path} :: neither old nor new found")
    return states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="custom_rasterizer tree (default: vendor/hunyuan3d-cuda/hy3dpaint/custom_rasterizer)",
    )
    parser.add_argument("--check", action="store_true", help="Report only; do not write")
    args = parser.parse_args(argv)

    root = args.root
    if not root.is_dir():
        print(f"MISSING root {root}")
        return 2

    states = check_tree(root) if args.check else apply_to_tree(root)
    print("\n".join(states))
    if args.check:
        return 0 if all(s.startswith("APPLIED") for s in states) else 2
    return 0 if all(s.startswith(("APPLIED", "PATCHED")) for s in states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
