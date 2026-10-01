#!/usr/bin/env python3
"""Fix MSVC C2398 narrowing in o-voxel torch shape brace-inits (size_t → int64_t).

On Windows, `{N, C}` where N/C are `size_t` (or `vector::size()`) is a narrowing
conversion into libtorch's `int64_t` dims. Cast explicitly so CUDA TRELLIS.2's
vendored o-voxel builds under VS 2022/18.

    python scripts/patch_ovoxel_msvc_narrowing.py
    python scripts/patch_ovoxel_msvc_narrowing.py --root vendor/trellis2-cuda/o-voxel
    python scripts/patch_ovoxel_msvc_narrowing.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis2-cuda" / "o-voxel"

# (relative path, old snippet, new snippet) — exact upstream text, idempotent via "new in file".
REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "src/io/filter_neighbor.cpp",
        "torch::Tensor delta = torch::zeros({N, C}, torch::dtype(torch::kUInt8));",
        "torch::Tensor delta = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)}, torch::dtype(torch::kUInt8));",
    ),
    (
        "src/io/filter_neighbor.cpp",
        "torch::Tensor attr = torch::zeros({N, C}, torch::dtype(torch::kUInt8));",
        "torch::Tensor attr = torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)}, torch::dtype(torch::kUInt8));",
    ),
    (
        "src/io/filter_parent.cpp",
        "torch::Tensor delta = torch::zeros({N_leaf, C}, torch::kUInt8);",
        "torch::Tensor delta = torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}, torch::kUInt8);",
    ),
    (
        "src/io/filter_parent.cpp",
        "torch::Tensor attr = torch::zeros({N_leaf, C}, torch::kUInt8);",
        "torch::Tensor attr = torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}, torch::kUInt8);",
    ),
    (
        "src/io/svo.cpp",
        "torch::Tensor svo_tensor = torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8).clone();",
        "torch::Tensor svo_tensor = torch::from_blob(svo.data(), {static_cast<int64_t>(svo.size())}, torch::kUInt8).clone();",
    ),
    (
        "src/io/svo.cpp",
        "torch::Tensor codes_tensor = torch::from_blob(codes.data(), {codes.size()}, torch::kInt32).clone();",
        "torch::Tensor codes_tensor = torch::from_blob(codes.data(), {static_cast<int64_t>(codes.size())}, torch::kInt32).clone();",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "torch::from_blob(voxels.data(), {int(voxels .size()), 3}, torch::kInt32).clone(),",
        "torch::from_blob(voxels.data(), {static_cast<int64_t>(voxels.size()), static_cast<int64_t>(3)}, torch::kInt32).clone(),",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "torch::from_blob(dual_vertices.data(), {int(dual_vertices.size()), 3}, torch::kFloat32).clone(),",
        "torch::from_blob(dual_vertices.data(), {static_cast<int64_t>(dual_vertices.size()), static_cast<int64_t>(3)}, torch::kFloat32).clone(),",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "torch::from_blob(intersected.data(), {int(intersected.size()), 3}, torch::kBool).clone()",
        "torch::from_blob(intersected.data(), {static_cast<int64_t>(intersected.size()), static_cast<int64_t>(3)}, torch::kBool).clone()",
    ),
)


def patch_source(source: str, old: str, new: str) -> tuple[str, bool]:
    """Apply one replacement. Returns (text, changed). Idempotent if already patched."""
    if new in source:
        return source, False
    if old not in source:
        raise RuntimeError(f"anchor missing for MSVC narrowing patch:\n  {old}")
    return source.replace(old, new, 1), True


def apply_to_tree(root: Path) -> list[str]:
    """Patch every known site under an o-voxel checkout. Returns status lines."""
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
        if new in text:
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
        help="o-voxel tree to patch (default: vendor/trellis2-cuda/o-voxel)",
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
