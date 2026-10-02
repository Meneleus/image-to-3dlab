#!/usr/bin/env python3
"""Fix basicsr vs modern torchvision so ``realesrgan`` can import.

``realesrgan`` pulls ``basicsr``, which still does::

    from torchvision.transforms.functional_tensor import rgb_to_grayscale

``functional_tensor`` was removed in torchvision 0.17+ (current torch CUDA
wheels). Point the import at ``torchvision.transforms.functional`` instead.
Idempotent.

Find the file via the Hunyuan vendor interpreter (or any ``--python``)::

    python scripts/patch_basicsr_functional_tensor.py \\
        --python vendor/hunyuan3d-cuda/.venv/Scripts/python.exe
    python scripts/patch_basicsr_functional_tensor.py --path path/to/degradations.py
    python scripts/patch_basicsr_functional_tensor.py --check --python …
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"
MARKER = "image-to-3dlab: torchvision.transforms.functional rgb_to_grayscale"


def apply(source: str) -> str:
    if OLD not in source:
        if NEW in source:
            return source
        raise SystemExit(
            "anchor not found: expected basicsr functional_tensor import in degradations.py"
        )
    # Keep a one-line note beside the fix for --check / humans.
    replacement = f"{NEW}  # {MARKER}"
    if MARKER in source:
        return source
    return source.replace(OLD, replacement, 1)


def is_patched(source: str) -> bool:
    return OLD not in source and NEW in source


def degradations_via_python(python: Path) -> Path:
    """Locate ``basicsr/data/degradations.py`` inside ``python``'s site-packages."""
    code = (
        "import basicsr, pathlib; "
        "print(pathlib.Path(basicsr.__file__).resolve().parent / 'data' / 'degradations.py')"
    )
    try:
        out = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not probe basicsr with {python}: {exc}") from exc
    if out.returncode != 0:
        err = (out.stderr or out.stdout or "").strip()
        raise SystemExit(
            f"basicsr not importable in {python} (install realesrgan first): {err}"
        )
    path = Path(out.stdout.strip())
    if not path.is_file():
        raise SystemExit(f"degradations.py missing at {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path,
                        help="Interpreter whose site-packages holds basicsr")
    parser.add_argument("--path", type=Path,
                        help="Direct path to basicsr/data/degradations.py")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.path:
        path = args.path
    elif args.python:
        path = degradations_via_python(args.python)
    else:
        parser.error("pass --python or --path")
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
