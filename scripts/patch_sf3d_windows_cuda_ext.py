#!/usr/bin/env python3
"""Make SF3D ``texture_baker`` / ``uv_unwrapper`` compile on Windows MSVC + CUDA 13.x.

Upstream ``texture_baker/setup.py`` always passes Linux gcc flags (``-O3``,
``-fopenmp``, ``-march=native``) and only appends a few MSVC flags inside
``if debug_mode`` — so a normal release build never gets
``/Zc:preprocessor``. CUDA 13 CCCL then aborts with C1189.

This replaces those flag blocks with MSVC-safe cxx lists and nvcc host flags
(``/std:c++20``, ``/Zc:preprocessor`` via ``-Xcompiler=``). Idempotent.

    python scripts/patch_sf3d_windows_cuda_ext.py
    python scripts/patch_sf3d_windows_cuda_ext.py --root vendor/stable-fast-3d
    python scripts/patch_sf3d_windows_cuda_ext.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "stable-fast-3d"

# Exact upstream texture_baker/setup.py (Windows append is inside debug_mode).
TEXTURE_BAKER_OLD = '''\
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-fopenmp",
        ]
        + ["-march=native"]
        if use_native_arch
        else [],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        if platform.system() == "Windows":
            extra_compile_args["cxx"].append("/Z7")
            extra_compile_args["cxx"].append("/Od")
            extra_link_args.extend(["/DEBUG"])
        extra_compile_args["cxx"].append("-UNDEBUG")
        extra_compile_args["nvcc"].append("-UNDEBUG")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])
'''

TEXTURE_BAKER_NEW = '''\
    # image-to-3dlab: MSVC-safe flags (CUDA 13 CCCL needs /Zc:preprocessor)
    if platform.system() == "Windows":
        extra_compile_args = {
            "cxx": [
                "/O2" if not debug_mode else "/Od",
                "/std:c++20",
                "/EHsc",
                "/Zc:preprocessor",
                "/Zc:__cplusplus",
                "/openmp",
            ],
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
                "-std=c++20",
                "-Xcompiler=/std:c++20",
                "-Xcompiler=/Zc:preprocessor",
                "-Xcompiler=/Zc:__cplusplus",
                "-allow-unsupported-compiler",
            ],
        }
        if debug_mode:
            extra_compile_args["cxx"].append("/Z7")
            extra_link_args.extend(["/DEBUG"])
    else:
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
                "-fopenmp",
            ]
            + (["-march=native"] if use_native_arch else []),
            "nvcc": [
                "-O3" if not debug_mode else "-O0",
            ],
        }
        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["cxx"].append("-UNDEBUG")
            extra_compile_args["nvcc"].append("-UNDEBUG")
            extra_compile_args["nvcc"].append("-g")
            extra_link_args.extend(["-O0", "-g"])
'''

UV_UNWRAPPER_OLD = '''\
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            ("-Xclang " if is_mac else "") + "-fopenmp",
        ]
        + ["-march=native"]
        if use_native_arch
        else []
        + ["-mmacosx-version-min=10.15"] if is_mac else [],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-UNDEBUG")
        extra_link_args.extend(["-O0", "-g"])
'''

UV_UNWRAPPER_NEW = '''\
    # image-to-3dlab: MSVC-safe cxx flags on Windows
    import platform as _i2l_platform
    if _i2l_platform.system() == "Windows":
        extra_compile_args = {
            "cxx": [
                "/O2" if not debug_mode else "/Od",
                "/std:c++20",
                "/EHsc",
                "/Zc:preprocessor",
                "/Zc:__cplusplus",
                "/openmp",
            ],
        }
        if debug_mode:
            extra_compile_args["cxx"].append("/Z7")
            extra_link_args.extend(["/DEBUG"])
    else:
        extra_compile_args = {
            "cxx": [
                "-O3" if not debug_mode else "-O0",
                "-fdiagnostics-color=always",
                ("-Xclang " if is_mac else "") + "-fopenmp",
            ]
            + (["-march=native"] if use_native_arch else [])
            + (["-mmacosx-version-min=10.15"] if is_mac else []),
        }
        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["cxx"].append("-UNDEBUG")
            extra_link_args.extend(["-O0", "-g"])
'''

REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("texture_baker/setup.py", TEXTURE_BAKER_OLD, TEXTURE_BAKER_NEW),
    ("uv_unwrapper/setup.py", UV_UNWRAPPER_OLD, UV_UNWRAPPER_NEW),
)

MARKER = "image-to-3dlab: MSVC-safe"


def patch_source(source: str, old: str, new: str) -> tuple[str, bool]:
    if MARKER in source and old not in source:
        return source, False
    if old not in source:
        if MARKER in source:
            return source, False
        raise RuntimeError(f"anchor missing for SF3D Windows CUDA patch:\n  {old[:80]}…")
    return source.replace(old, new, 1), True


def apply_to_tree(root: Path) -> list[str]:
    states: list[str] = []
    for rel, old, new in REPLACEMENTS:
        path = root / rel
        if not path.is_file():
            states.append(f"MISSING {path}")
            continue
        text = path.read_text(encoding="utf-8")
        text, changed = patch_source(text, old, new)
        if changed:
            path.write_text(text, encoding="utf-8")
            states.append(f"PATCHED {path}")
        else:
            states.append(f"APPLIED {path}")
    return states


def check_tree(root: Path) -> list[str]:
    states: list[str] = []
    for rel, old, new in REPLACEMENTS:
        path = root / rel
        if not path.is_file():
            states.append(f"MISSING {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text and old not in text:
            states.append(f"APPLIED {path}")
        elif old in text:
            states.append(f"ABSENT {path}")
        else:
            states.append(f"UNKNOWN {path}")
    return states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"MISSING root {args.root}")
        return 2
    states = check_tree(args.root) if args.check else apply_to_tree(args.root)
    print("\n".join(states))
    if args.check:
        return 0 if all(s.startswith("APPLIED") for s in states) else 2
    return 0 if all(s.startswith(("APPLIED", "PATCHED")) for s in states) else 2


if __name__ == "__main__":
    raise SystemExit(main())
