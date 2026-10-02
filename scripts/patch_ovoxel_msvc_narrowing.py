#!/usr/bin/env python3
"""Fix MSVC build breaks in vendored o-voxel (narrowing + invalid `d` float suffixes).

On Windows / VS 2022–18:

* `{N, C}` with `size_t` into libtorch `int64_t` dims → C2398 (cast to int64_t).
* Upstream `1e-6d` / `0.0d` are not MSVC double literals under C++20 → C3688
  (`operator ""d` not found). Plain `1e-6` / `0.0` are already double.
* `int4{i, neigh_indices[…]}` with `size_t` neighbours → C4838 (cast to int).

Only exact anchors are replaced; re-running is idempotent.

    python scripts/patch_ovoxel_msvc_narrowing.py
    python scripts/patch_ovoxel_msvc_narrowing.py --root vendor/trellis2-cuda/o-voxel
    python scripts/patch_ovoxel_msvc_narrowing.py --check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis2-cuda" / "o-voxel"

# Upstream GCC-style double suffixes MSVC rejects under C++20 (C3688).
# Patched text must never invent these; we only strip known anchors.
_BAD_DOUBLE_SUFFIX = re.compile(
    r"(?<![\w.])(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?d\b"
)


def _assert_no_invented_d_suffix(text: str, rel: str) -> None:
    """Fail loudly if a patch left MSVC-invalid `…d` float literals behind."""
    hits = _BAD_DOUBLE_SUFFIX.findall(text)
    if hits:
        raise RuntimeError(
            f"{rel}: still has MSVC-invalid float suffix(es) {hits!r} "
            "(would become C3688 / operator \"\"d)"
        )


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
    # --- flexible_dual_grid.cpp: MSVC rejects upstream `…d` float suffixes (C3688) ---
    (
        "src/convert/flexible_dual_grid.cpp",
        "if (segment_length < 1e-6d) continue; // Skip degenerate edges (zero-length)",
        "if (segment_length < 1e-6) continue; // Skip degenerate edges (zero-length)",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "if (dir[axis] == 0.0d) {",
        "if (dir[axis] == 0.0) {",
    ),
    # --- flexible_dual_grid.cpp: size_t → int in int4 brace-init (C4838) ---
    (
        "src/convert/flexible_dual_grid.cpp",
        "int4 quad_indices{i, neigh_indices[0], neigh_indices[2], neigh_indices[1]};",
        "int4 quad_indices{i, static_cast<int>(neigh_indices[0]), static_cast<int>(neigh_indices[2]), static_cast<int>(neigh_indices[1])};",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "int4 quad_indices{i, neigh_indices[1], neigh_indices[5], neigh_indices[3]};",
        "int4 quad_indices{i, static_cast<int>(neigh_indices[1]), static_cast<int>(neigh_indices[5]), static_cast<int>(neigh_indices[3])};",
    ),
    (
        "src/convert/flexible_dual_grid.cpp",
        "int4 quad_indices{i, neigh_indices[0], neigh_indices[4], neigh_indices[3]};",
        "int4 quad_indices{i, static_cast<int>(neigh_indices[0]), static_cast<int>(neigh_indices[4]), static_cast<int>(neigh_indices[3])};",
    ),
    # --- flexible_dual_grid.cpp: torch from_blob shapes (C2398) ---
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
    """Apply one replacement. Returns (text, changed). Idempotent if already patched.

    Prefer the already-patched form when both could match as substrings of each
    other (e.g. stripping a trailing ``d`` from a float literal).
    """
    if new in source and old not in source:
        return source, False
    if new in source and old in source:
        # Already partially patched elsewhere; still replace this exact old once.
        return source.replace(old, new, 1), True
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
        if rel.endswith("flexible_dual_grid.cpp"):
            _assert_no_invented_d_suffix(text, rel)
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
