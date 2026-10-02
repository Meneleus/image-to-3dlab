#!/usr/bin/env python3
"""Fix Hunyuan paint mesh simplify for modern trimesh.

Upstream ``hy3dpaint/utils/simplify_mesh_utils.py`` calls::

    courent.simplify_quadric_decimation(target_count)

Newer trimesh treats the first positional as ``percent`` (0–1). A face count
like 40000 then becomes ``target_reduction`` for ``fast_simplification`` and
raises ``ValueError: target_reduction must be between 0 and 1``. Pass
``face_count=`` instead; fall back to the positional form for older trimesh.
Idempotent.

    python scripts/patch_hunyuan_simplify_face_count.py
    python scripts/patch_hunyuan_simplify_face_count.py \\
        --root vendor/hunyuan3d-cuda/hy3dpaint/utils
    python scripts/patch_hunyuan_simplify_face_count.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "hunyuan3d-cuda" / "hy3dpaint" / "utils"
TARGET = "simplify_mesh_utils.py"
MARKER = "image-to-3dlab: face_count= for modern trimesh"

OLD = "        courent = courent.simplify_quadric_decimation(target_count)\n"

NEW = """\
        # image-to-3dlab: face_count= for modern trimesh
        try:
            courent = courent.simplify_quadric_decimation(face_count=target_count)
        except TypeError:
            # Older trimesh used a positional face-count argument.
            courent = courent.simplify_quadric_decimation(target_count)
"""


def apply(source: str) -> str:
    # Marker alone — OLD is a substring of the TypeError fallback indent.
    if MARKER in source:
        return source
    if "face_count=target_count" in source:
        return source
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line == OLD:
            # NEW is multi-line; keep trailing newline on the block.
            replacement = NEW if NEW.endswith("\n") else NEW + "\n"
            return "".join(lines[:i]) + replacement + "".join(lines[i + 1 :])
    raise SystemExit(
        "anchor not found: simplify_quadric_decimation(target_count) in "
        "simplify_mesh_utils.py"
    )


def is_patched(source: str) -> bool:
    return MARKER in source and "face_count=target_count" in source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = args.root / TARGET if args.root.is_dir() else args.root
    if not path.is_file():
        print(f"MISSING {path}")
        return 2
    text = path.read_text(encoding="utf-8")
    if args.check:
        print(f"{'APPLIED' if is_patched(text) else 'ABSENT'} {path}")
        return 0 if is_patched(text) else 2
    new = apply(text)
    if new == text:
        print(f"APPLIED {path}")
        return 0
    path.write_text(new, encoding="utf-8")
    print(f"PATCHED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
