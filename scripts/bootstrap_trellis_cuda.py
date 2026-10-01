#!/usr/bin/env python3
"""Install official TRELLIS.2 for NVIDIA (Linux or Windows) into vendor/trellis2-cuda.

Clones https://github.com/microsoft/TRELLIS.2, creates a Python 3.11 venv with CUDA
PyTorch, installs the basic Python deps, then builds the CUDA extensions the pipeline
needs (nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel). flash-attn is best-effort:
when it cannot build, the generate wrapper falls back to PyTorch SDPA.

Microsoft's README only claims Linux testing. On Windows the same steps run when the
CUDA toolkit and Visual Studio C++ build tools are present; if an extension fails to
compile, this says so and stops without claiming success.

On Windows this forces C++20 end-to-end: rewrites extension `setup.py`, patches the
venv's `torch.utils.cpp_extension` (older wheels hardcode `-std=c++17` into ninja),
injects a runtime sanitizer so cl/nvcc lines never mix 17 and 20, and wipes stale
build dirs. Env only carries `/Zc:preprocessor` — not a second `/std:` (that causes
cl D9025 flip-flops with `--use-local-env`).

Weights (~14 GB TRELLIS.2-4B + gated DINOv3) are *not* fetched here — they arrive on
the first generation run, same as the Mac bootstrap. BRIA RMBG-2.0 is never installed.

`AGENTS.md`: name the backend, route and size, and require a yes (or `--yes`).

    python scripts/bootstrap_trellis_cuda.py
    python scripts/bootstrap_trellis_cuda.py --yes
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host
from image_to_3dlab.cuda_routes import TRELLIS_CUDA, venv_python

UPSTREAM = "https://github.com/microsoft/TRELLIS.2.git"
VENDOR = TRELLIS_CUDA

# Rough install footprint excluding weights (clone + compiled extensions).
CODE_GB = 2.5

EXTENSIONS = (
    ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0"),
    ("nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git", "renderutils"),
    ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", None),
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", None),
)

BASIC = [
    "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
    "ninja", "trimesh", "transformers", "tensorboard", "pandas", "lpips",
    "zstandard", "kornia", "timm", "Pillow", "huggingface_hub", "rembg",
    "fast_simplification",
]

# CUDA 12.8+/13.x CCCL + current PyTorch headers need C++20 and MSVC's conformant
# preprocessor. Sources of c++17 on Windows:
#   1) FlexGEMM/CuMesh setup.py hardcodes /std:c++17
#   2) Older torch.utils.cpp_extension hardcodes -std=c++17 into ninja cuda_cflags
#   3) Stale build/ ninja files from a previous failed compile
# Putting /std:c++20 in CL *and* on the command line causes D9025 flip-flops under
# nvcc --use-local-env. So env carries only Zc flags; std comes from setup.py + torch.
WIN_CL_FLAGS = "/Zc:preprocessor /Zc:__cplusplus"
WIN_NVCC_FLAGS = (
    "-allow-unsupported-compiler "
    "-Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus"
)
WIN_SETUP_MARKER = "# image-to-3dlab: windows cxx20 + Zc:preprocessor\n"
WIN_HOOK_BEGIN = "# --- image-to-3dlab: force C++20 on cl/nvcc (begin) ---\n"
WIN_HOOK_END = "# --- image-to-3dlab: force C++20 on cl/nvcc (end) ---\n"
WIN_HOOK_ENSURE_NAME = "_i2l_ensure_msvc_cccl_flags"
_CXX17_IN_SETUP = re.compile(r"c\+\+17", re.IGNORECASE)

# Flags every CUDAExtension must get on Windows (CCCL + current PyTorch headers).
WIN_MSVC_CXX_FLAGS = ("/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus")
WIN_MSVC_NVCC_FLAGS = (
    "-Xcompiler=/std:c++20",
    "-Xcompiler=/Zc:preprocessor",
    "-Xcompiler=/Zc:__cplusplus",
)

# Injected into *every* extension setup.py the bootstrap builds. Sanitizes c++17,
# appends CCCL flags to cxx/nvcc lists (nvdiffrast/nvdiffrec have none upstream),
# and strips mixed standards from ninja/spawn command lines.
WIN_CXX20_HOOK = '''\
# --- image-to-3dlab: force C++20 on cl/nvcc (begin) ---
def _i2l_sanitize_std_flags(flags):
    """Replace c++17→c++20; keep at most one /std:, -std=c++*, -Xcompiler=/std:."""
    if not flags:
        return flags
    result = []
    msvc_std = gnu_std = xcomp_std = None
    i, n = 0, len(flags)
    while i < n:
        f = flags[i]
        if not isinstance(f, str):
            result.append(f)
            i += 1
            continue
        f = f.replace("c++17", "c++20").replace("C++17", "c++20")
        if f.startswith("/std:"):
            msvc_std = "/std:c++20"
            i += 1
            continue
        if f.startswith("-std=") and "c++" in f.lower():
            gnu_std = "-std=c++20"
            i += 1
            continue
        if f.startswith(("-Xcompiler=/std:", "-Xcompiler,/std:")):
            xcomp_std = "-Xcompiler=/std:c++20"
            i += 1
            continue
        if f == "-Xcompiler" and i + 1 < n and isinstance(flags[i + 1], str):
            nxt = flags[i + 1].replace("c++17", "c++20").replace("C++17", "c++20")
            if nxt.startswith("/std:"):
                xcomp_std = "-Xcompiler=/std:c++20"
                i += 2
                continue
        result.append(f)
        i += 1
    if msvc_std:
        result.append(msvc_std)
    if gnu_std:
        result.append(gnu_std)
    if xcomp_std:
        result.append(xcomp_std)
    return result

def _i2l_ensure_msvc_cccl_flags(extra_compile_args):
    """Append /std:c++20 + /Zc:preprocessor to every CUDAExtension on Windows."""
    cxx_need = ["/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus"]
    nvcc_need = [
        "-Xcompiler=/std:c++20",
        "-Xcompiler=/Zc:preprocessor",
        "-Xcompiler=/Zc:__cplusplus",
    ]
    if isinstance(extra_compile_args, dict):
        cxx = _i2l_sanitize_std_flags(list(extra_compile_args.get("cxx") or []))
        nvcc = _i2l_sanitize_std_flags(list(extra_compile_args.get("nvcc") or []))
        for flag in cxx_need:
            if flag not in cxx:
                cxx.append(flag)
        for flag in nvcc_need:
            if flag not in nvcc:
                nvcc.append(flag)
        extra_compile_args["cxx"] = cxx
        extra_compile_args["nvcc"] = nvcc
        return extra_compile_args
    if isinstance(extra_compile_args, (list, tuple)):
        flags = _i2l_sanitize_std_flags(list(extra_compile_args))
        for flag in cxx_need:
            if flag not in flags:
                flags.append(flag)
        return flags
    return {
        "cxx": list(cxx_need),
        "nvcc": list(nvcc_need),
    }

def _i2l_force_cxx20():
    try:
        import torch.utils.cpp_extension as _tce
    except ImportError:
        return
    _orig_write = _tce._write_ninja_file
    def _write_ninja_file(*args, **kwargs):
        args = list(args)
        for idx in range(1, min(5, len(args))):
            if args[idx] is not None:
                args[idx] = _i2l_sanitize_std_flags(list(args[idx]))
        for key in ("cflags", "post_cflags", "cuda_cflags", "cuda_post_cflags"):
            if kwargs.get(key) is not None:
                kwargs[key] = _i2l_sanitize_std_flags(list(kwargs[key]))
        return _orig_write(*args, **kwargs)
    _tce._write_ninja_file = _write_ninja_file
    _orig_be = _tce.BuildExtension.build_extensions
    def build_extensions(self):
        for ext in self.extensions:
            eca = getattr(ext, "extra_compile_args", None)
            ext.extra_compile_args = _i2l_ensure_msvc_cccl_flags(
                dict(eca) if isinstance(eca, dict) else eca
            )
        compiler = self.compiler
        if compiler is not None and not getattr(compiler, "_i2l_cxx20_wrapped", False):
            _spawn = compiler.spawn
            def spawn(cmd, *a, **k):
                if isinstance(cmd, (list, tuple)):
                    cmd = _i2l_sanitize_std_flags(list(cmd))
                return _spawn(cmd, *a, **k)
            compiler.spawn = spawn
            compiler._i2l_cxx20_wrapped = True
        return _orig_be(self)
    _tce.BuildExtension.build_extensions = build_extensions

_i2l_force_cxx20()
# --- image-to-3dlab: force C++20 on cl/nvcc (end) ---
'''


def announcement() -> str:
    family = host.os_family()
    lines = [
        "",
        "About to install:",
        "",
        "  backend: TRELLIS.2 (official CUDA)",
        f"  route:   clone + CUDA extensions into {VENDOR.relative_to(REPO)}/",
        f"  code:    ~{CODE_GB:.1f} GB (no model weights yet)",
        "  weights: ~14 GB TRELLIS.2-4B + gated DINOv3 on first Generate run",
        f"  host:    {family} NVIDIA",
        "",
        "  licence: MIT (code + weights); DINOv3 encoder is separately gated",
        "           https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "",
        "  note:    Microsoft tests TRELLIS.2 on Linux. Windows needs the CUDA",
        "           toolkit and Visual Studio C++ build tools for the extensions.",
    ]
    if family == "windows":
        lines += [
            "           Forces C++20 (setup.py + torch cpp_extension + sanitizer)",
            "           and /Zc:preprocessor for CUDA 12.8+/13.x + modern MSVC.",
        ]
    lines.append("")
    return "\n".join(lines)


def windows_cuda_build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for building CUDA extensions on Windows with CUDA 12.8+/13.x + MSVC.

    Sets DISTUTILS_USE_SDK and /Zc:preprocessor. Does *not* put /std:c++20 in CL —
    that races with torch/setup.py under nvcc --use-local-env (cl D9025 flip-flops).
    No-op on non-Windows hosts.
    """
    env = dict(base if base is not None else os.environ)
    if host.os_family() != "windows":
        return env
    env["DISTUTILS_USE_SDK"] = "1"
    # Strip any user-exported /std: from CL/CXXFLAGS so only one standard remains.
    env["CL"] = WIN_CL_FLAGS
    env["CXXFLAGS"] = WIN_CL_FLAGS
    env["NVCC_FLAGS"] = WIN_NVCC_FLAGS
    env["NVCC_PREPEND_FLAGS"] = WIN_NVCC_FLAGS
    return env


def setup_text_outside_hook(text: str) -> str:
    """setup.py with our injected runtime hook removed (for validation / rewrites)."""
    if WIN_HOOK_BEGIN in text and WIN_HOOK_END in text:
        start = text.index(WIN_HOOK_BEGIN)
        end = text.index(WIN_HOOK_END) + len(WIN_HOOK_END)
        return text[:start] + text[end:]
    return text


def setup_still_has_cxx17(text: str) -> bool:
    """True if setup.py still names C++17 outside our injected hook comments."""
    return _CXX17_IN_SETUP.search(setup_text_outside_hook(text)) is not None


def setup_has_cccl_ensure_hook(text: str) -> bool:
    """True when the runtime hook that appends CCCL flags to every extension is present."""
    return (
        WIN_HOOK_BEGIN in text
        and WIN_HOOK_END in text
        and WIN_HOOK_ENSURE_NAME in text
        and "/Zc:preprocessor" in text
        and "/std:c++20" in text
    )


def sanitize_std_flags(flags: list) -> list:
    """Replace c++17→c++20; keep at most one /std:, -std=c++*, -Xcompiler=/std:.

    Pure helper (same rules as the setup.py runtime hook). Compile lines must never
    carry both c++17 and c++20.
    """
    if not flags:
        return flags
    result: list = []
    msvc_std = gnu_std = xcomp_std = None
    i, n = 0, len(flags)
    while i < n:
        f = flags[i]
        if not isinstance(f, str):
            result.append(f)
            i += 1
            continue
        f = f.replace("c++17", "c++20").replace("C++17", "c++20")
        if f.startswith("/std:"):
            msvc_std = "/std:c++20"
            i += 1
            continue
        if f.startswith("-std=") and "c++" in f.lower():
            gnu_std = "-std=c++20"
            i += 1
            continue
        if f.startswith(("-Xcompiler=/std:", "-Xcompiler,/std:")):
            xcomp_std = "-Xcompiler=/std:c++20"
            i += 1
            continue
        if f == "-Xcompiler" and i + 1 < n and isinstance(flags[i + 1], str):
            nxt = flags[i + 1].replace("c++17", "c++20").replace("C++17", "c++20")
            if nxt.startswith("/std:"):
                xcomp_std = "-Xcompiler=/std:c++20"
                i += 2
                continue
        result.append(f)
        i += 1
    if msvc_std:
        result.append(msvc_std)
    if gnu_std:
        result.append(gnu_std)
    if xcomp_std:
        result.append(xcomp_std)
    return result


def ensure_msvc_cccl_flags(extra_compile_args: dict | list | None) -> dict | list:
    """Append /std:c++20 + /Zc:preprocessor to CUDAExtension args (testable mirror of hook)."""
    cxx_need = list(WIN_MSVC_CXX_FLAGS)
    nvcc_need = list(WIN_MSVC_NVCC_FLAGS)
    if isinstance(extra_compile_args, dict):
        out = dict(extra_compile_args)
        cxx = sanitize_std_flags(list(out.get("cxx") or []))
        nvcc = sanitize_std_flags(list(out.get("nvcc") or []))
        for flag in cxx_need:
            if flag not in cxx:
                cxx.append(flag)
        for flag in nvcc_need:
            if flag not in nvcc:
                nvcc.append(flag)
        out["cxx"] = cxx
        out["nvcc"] = nvcc
        return out
    if isinstance(extra_compile_args, (list, tuple)):
        flags = sanitize_std_flags(list(extra_compile_args))
        for flag in cxx_need:
            if flag not in flags:
                flags.append(flag)
        return flags
    return {"cxx": cxx_need, "nvcc": nvcc_need}


def inject_cxx20_runtime_hook(text: str) -> str:
    """Ensure the CCCL/C++20 runtime hook is present (replace older hook revisions)."""
    if setup_has_cccl_ensure_hook(text):
        return text
    if WIN_HOOK_BEGIN in text and WIN_HOOK_END in text:
        start = text.index(WIN_HOOK_BEGIN)
        end = text.index(WIN_HOOK_END) + len(WIN_HOOK_END)
        return text[:start] + WIN_CXX20_HOOK + text[end:]
    future = "from __future__ import annotations\n"
    if future in text:
        return text.replace(future, future + "\n" + WIN_CXX20_HOOK + "\n", 1)
    return WIN_CXX20_HOOK + "\n" + text


def _ensure_zc_preprocessor(text: str) -> str:
    """Add /Zc:preprocessor on MSVC cxx and nvcc -Xcompiler lines when missing.

    Operates on the full text; callers should pass setup_text_outside_hook when the
    runtime hook is already present so hook string literals are not mistaken for flags.
    """
    if "/Zc:preprocessor" in text:
        return text
    if "/Zc:__cplusplus" in text:
        text = text.replace(
            '"/Zc:__cplusplus"',
            '"/Zc:__cplusplus", "/Zc:preprocessor"',
        )
        text = text.replace(
            "'/Zc:__cplusplus'",
            "'/Zc:__cplusplus', '/Zc:preprocessor'",
        )
        text = text.replace(
            '"-Xcompiler=/Zc:__cplusplus"',
            '"-Xcompiler=/Zc:__cplusplus", "-Xcompiler=/Zc:preprocessor"',
        )
        text = text.replace(
            "'-Xcompiler=/Zc:__cplusplus'",
            "'-Xcompiler=/Zc:__cplusplus', '-Xcompiler=/Zc:preprocessor'",
        )
        return text
    if '"/std:c++20"' in text:
        text = text.replace(
            '"/std:c++20"',
            '"/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus"',
            1,
        )
    if '"-Xcompiler=/std:c++20"' in text:
        text = text.replace(
            '"-Xcompiler=/std:c++20"',
            '"-Xcompiler=/std:c++20", "-Xcompiler=/Zc:preprocessor", '
            '"-Xcompiler=/Zc:__cplusplus"',
            1,
        )
    return text


def _rewrite_unix_only_args_for_msvc(text: str) -> str:
    """Turn unix-only cxx/nvcc lists (o-voxel) into MSVC-friendly Windows flags."""
    if "/std:c++" in text:
        return text
    text = re.sub(
        r'"cxx":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]',
        '"cxx": ["/O2", "/std:c++20", "/EHsc", "/Zc:__cplusplus", "/Zc:preprocessor"]',
        text,
    )
    text = re.sub(
        r'"nvcc":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]\s*\+\s*cc_flag',
        (
            '"nvcc": ["-O3", "-std=c++20", "-Xcompiler=/std:c++20", '
            '"-Xcompiler=/EHsc", "-Xcompiler=/Zc:__cplusplus", '
            '"-Xcompiler=/Zc:preprocessor", "-allow-unsupported-compiler"] + cc_flag'
        ),
        text,
    )
    text = re.sub(
        r'"nvcc":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]',
        (
            '"nvcc": ["-O3", "-std=c++20", "-Xcompiler=/std:c++20", '
            '"-Xcompiler=/EHsc", "-Xcompiler=/Zc:__cplusplus", '
            '"-Xcompiler=/Zc:preprocessor", "-allow-unsupported-compiler"]'
        ),
        text,
    )
    return text


def _inject_msvc_cccl_source_flags(text: str) -> str:
    """Add MSVC CCCL flags into known setup.py shapes that omit them (nvdiffrast, …).

    The runtime hook always appends these too; this makes them visible in the file
    and covers packages whose Windows branch only lists warning suppressions.
    """
    if (
        all(f'"{flag}"' in text or f"'{flag}'" in text for flag in WIN_MSVC_CXX_FLAGS)
        and all(
            f'"{flag}"' in text or f"'{flag}'" in text for flag in WIN_MSVC_NVCC_FLAGS
        )
    ):
        return text

    # nvdiffrast: Windows warning suppressions list — append CCCL cxx flags.
    old_nvd = '["/wd4067", "/wd4624", "/wd4996"]'
    new_nvd = (
        '["/wd4067", "/wd4624", "/wd4996", '
        '"/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus"]'
    )
    if old_nvd in text and "/Zc:preprocessor" not in setup_text_outside_hook(text):
        text = text.replace(old_nvd, new_nvd, 1)

    # nvdiffrast nvcc list has no host std/Zc — extend it.
    old_nvcc = '"nvcc": ["-DNVDR_TORCH", "-lineinfo"]'
    new_nvcc = (
        '"nvcc": ["-DNVDR_TORCH", "-lineinfo", '
        '"-Xcompiler=/std:c++20", "-Xcompiler=/Zc:preprocessor", '
        '"-Xcompiler=/Zc:__cplusplus"]'
    )
    if old_nvcc in text and "-Xcompiler=/Zc:preprocessor" not in setup_text_outside_hook(text):
        text = text.replace(old_nvcc, new_nvcc, 1)

    # nvdiffrec: bare c_flags / nvcc_flags — append on Windows.
    marker = "# image-to-3dlab: msvc cccl flags for nvdiffrec\n"
    if (
        "c_flags = ['-DNVDR_TORCH']" in text
        and marker not in text
        and "/Zc:preprocessor" not in setup_text_outside_hook(text)
    ):
        text = text.replace(
            "c_flags = ['-DNVDR_TORCH']\n"
            "nvcc_flags = ['-DNVDR_TORCH']\n",
            "c_flags = ['-DNVDR_TORCH']\n"
            "nvcc_flags = ['-DNVDR_TORCH']\n"
            + marker
            + "if os.name == 'nt':\n"
            "    c_flags += ['/std:c++20', '/Zc:preprocessor', '/Zc:__cplusplus']\n"
            "    nvcc_flags += ['-Xcompiler=/std:c++20', '-Xcompiler=/Zc:preprocessor', "
            "'-Xcompiler=/Zc:__cplusplus']\n",
            1,
        )
    return text


def rewrite_setup_cxx_flags(text: str) -> str:
    """Replace every C++17 compile-flag spelling with C++20; add /Zc:preprocessor.

    Pure string rewrite used on Windows before building every CUDA extension.
    Must *replace* hardcoded c++17 — prepending c++20 via env loses (cl D9025).
    """
    hook = ""
    body = text
    if WIN_HOOK_BEGIN in text and WIN_HOOK_END in text:
        start = text.index(WIN_HOOK_BEGIN)
        end = text.index(WIN_HOOK_END) + len(WIN_HOOK_END)
        hook, body = text[start:end], text[:start] + text[end:]

    body = body.replace("-Xcompiler=/std:c++17", "-Xcompiler=/std:c++20")
    body = body.replace("/std:c++17", "/std:c++20")
    body = body.replace("-std=c++17", "-std=c++20")
    body = _CXX17_IN_SETUP.sub("c++20", body)
    body = _ensure_zc_preprocessor(body)
    body = _rewrite_unix_only_args_for_msvc(body)
    body = _inject_msvc_cccl_source_flags(body)

    if hook:
        return _rejoin_hook(body, hook)
    return body


def _rejoin_hook(body: str, hook: str) -> str:
    """Put the runtime hook back near the top of setup.py after body rewrites."""
    if WIN_HOOK_BEGIN in body:
        return body
    future = "from __future__ import annotations\n"
    if future in body:
        return body.replace(future, future + "\n" + hook + "\n", 1)
    if body.startswith(WIN_SETUP_MARKER):
        return WIN_SETUP_MARKER + hook + "\n" + body[len(WIN_SETUP_MARKER):]
    return hook + "\n" + body


def patch_windows_extension_setup(setup_py: Path) -> bool:
    """Rewrite every extension setup.py: c++17→c++20, CCCL flags, runtime hook.

    Used for nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, and o-voxel. Idempotent when
    the CCCL ensure hook is present and no c++17 remains outside it.
    """
    if host.os_family() != "windows":
        return False
    if not setup_py.is_file():
        return False
    text = setup_py.read_text(encoding="utf-8")
    if (
        not setup_still_has_cxx17(text)
        and WIN_SETUP_MARKER in text
        and setup_has_cccl_ensure_hook(text)
    ):
        return False
    original = text
    text = rewrite_setup_cxx_flags(text)
    text = inject_cxx20_runtime_hook(text)
    if WIN_SETUP_MARKER not in text:
        text = WIN_SETUP_MARKER + text
    if text == original:
        return False
    setup_py.write_text(text, encoding="utf-8")
    print(f"Patched Windows CUDA flags in {setup_py}", flush=True)
    return True


def ensure_windows_extension_setup(setup_py: Path) -> None:
    """Patch setup.py on Windows and refuse to build until CCCL/C++20 is wired in.

    Validates outside the hook body so `/std:c++20` string literals inside the
    sanitizer cannot fake a pass (that was the nvdiffrast false failure).
    """
    if host.os_family() != "windows":
        return
    if not setup_py.is_file():
        raise SystemExit(f"Missing {setup_py}; cannot apply Windows CUDA compile flags.")
    patch_windows_extension_setup(setup_py)
    text = setup_py.read_text(encoding="utf-8")
    if setup_still_has_cxx17(text):
        raise SystemExit(
            f"{setup_py} still contains c++17 after patching. "
            "Refusing to build — later /std:c++17 would override C++20 (cl D9025) "
            "and break PyTorch headers. See docs/WINDOWS.md."
        )
    if not setup_has_cccl_ensure_hook(text):
        raise SystemExit(
            f"{setup_py} is missing the CCCL ensure hook "
            f"({WIN_HOOK_ENSURE_NAME}) that appends /std:c++20 and "
            "/Zc:preprocessor to every CUDAExtension. See docs/WINDOWS.md."
        )
    print(f"Verified Windows CCCL/C++20 hook in {setup_py}", flush=True)


def patch_torch_cpp_extension_cxx20(py: Path) -> bool:
    """Rewrite c++17 → c++20 in this venv's torch.utils.cpp_extension.

    Torch ≤2.11 hardcodes `-std=c++17` / `/std:c++17` into the Windows ninja rules.
    That lands on the same cl/nvcc line as our C++20 flags (D9025). Idempotent.
    """
    if host.os_family() != "windows":
        return False
    probe = subprocess.run(
        [str(py), "-c", "import torch.utils.cpp_extension as m; print(m.__file__)"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"Cannot import torch.utils.cpp_extension with {py}:\n{probe.stderr}"
        )
    path = Path(probe.stdout.strip())
    text = path.read_text(encoding="utf-8")
    if "c++17" not in text:
        print(f"torch cpp_extension already c++20-only: {path}", flush=True)
        return False
    path.write_text(text.replace("c++17", "c++20"), encoding="utf-8")
    print(f"Patched torch cpp_extension c++17→c++20: {path}", flush=True)
    return True


def patch_ovoxel_msvc_narrowing(o_voxel: Path) -> None:
    """Apply MSVC o-voxel source fixes (C2398 / C3688 / C4838) before compile."""
    if host.os_family() != "windows":
        return
    script = REPO / "scripts" / "patch_ovoxel_msvc_narrowing.py"
    print(f"Patching o-voxel for MSVC narrowing ({script.name})...", flush=True)
    run([sys.executable, str(script), "--root", str(o_voxel)])


def clean_extension_build_artifacts(target: Path) -> None:
    """Remove stale setuptools/ninja outputs so old /std:c++17 rules cannot linger."""
    removed = False
    for name in ("build", "dist", ".eggs"):
        path = target / name
        if path.is_dir():
            shutil.rmtree(path)
            removed = True
            print(f"Removed stale build dir {path}", flush=True)
    for path in target.glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed = True
        print(f"Removed {path}", flush=True)
    for path in target.rglob("build.ninja"):
        path.unlink()
        removed = True
        print(f"Removed {path}", flush=True)
    if not removed:
        print(f"No stale build artifacts under {target}", flush=True)


def extension_build_env() -> dict[str, str]:
    """Full env for a pip install of a CUDA extension (PATH/CUDA_HOME + Windows flags)."""
    env = windows_cuda_build_env()
    nvcc = host.find_nvcc()
    if nvcc:
        env["PATH"] = os.pathsep.join([str(Path(nvcc).parent), env.get("PATH", "")])
        env.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
        env.setdefault("CUDA_PATH", str(Path(nvcc).parent.parent))
    return env


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_clone() -> None:
    if (VENDOR / ".git").is_dir():
        print(f"Already cloned: {VENDOR}", flush=True)
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--recursive", UPSTREAM, str(VENDOR)])


def ensure_venv() -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required. Install from https://docs.astral.sh/uv/")
    py = venv_python(VENDOR)
    if not py.is_file():
        run([uv, "venv", "--python", "3.11", str(VENDOR / ".venv")])
    return py


def install_torch(py: Path) -> None:
    uv = shutil.which("uv")
    index = host.torch_cuda_index(host.driver_cuda_version())
    if not index:
        raise SystemExit(
            "No PyTorch CUDA build matches this NVIDIA driver (need CUDA 12.8+). "
            "Update the driver, then run this again."
        )
    print(f"Installing PyTorch with CUDA from {index}", flush=True)
    run([uv, "pip", "install", "--python", str(py), "--index-url", index,
         "torch", "torchvision"])


def install_basic(py: Path) -> None:
    uv = shutil.which("uv")
    run([uv, "pip", "install", "--python", str(py), *BASIC])
    run([uv, "pip", "install", "--python", str(py),
         ("git+https://github.com/EasternJournalist/utils3d.git"
          "@9a4eb15e4021b67b12c460c7057d642626897ec8")])


def install_extension(py: Path, name: str, url: str, ref: str | None,
                      work: Path) -> None:
    target = work / name
    if not (target / ".git").is_dir():
        cmd = ["git", "clone", "--recursive", url, str(target)]
        run(cmd)
        if ref:
            run(["git", "checkout", ref], cwd=target)
    setup_py = target / "setup.py"
    ensure_windows_extension_setup(setup_py)
    clean_extension_build_artifacts(target)
    uv = shutil.which("uv")
    run([uv, "pip", "install", "--python", str(py), "--no-build-isolation",
         "--force-reinstall", "--no-deps", str(target)],
        env=extension_build_env())


def install_extensions(py: Path) -> None:
    if host.os_family() == "windows":
        print("\nPatching torch.utils.cpp_extension for C++20...", flush=True)
        patch_torch_cpp_extension_cxx20(py)
    work = VENDOR / ".i2l-build"
    work.mkdir(parents=True, exist_ok=True)
    for name, url, ref in EXTENSIONS:
        print(f"\nBuilding {name}...", flush=True)
        try:
            install_extension(py, name, url, ref, work)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Failed to build {name}. TRELLIS.2 needs a working CUDA toolkit "
                f"({'and Visual Studio C++ build tools ' if host.os_family() == 'windows' else ''}"
                f"matching this machine). See docs/WINDOWS.md.\n{exc}"
            ) from exc
    # o-voxel ships inside the TRELLIS.2 clone
    o_voxel = VENDOR / "o-voxel"
    if o_voxel.is_dir():
        print("\nBuilding o-voxel...", flush=True)
        ensure_windows_extension_setup(o_voxel / "setup.py")
        patch_ovoxel_msvc_narrowing(o_voxel)
        clean_extension_build_artifacts(o_voxel)
        uv = shutil.which("uv")
        try:
            run([uv, "pip", "install", "--python", str(py), "--no-build-isolation",
                 "--force-reinstall", "--no-deps", str(o_voxel)],
                env=extension_build_env())
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Failed to build o-voxel: {exc}") from exc


def try_flash_attn(py: Path) -> None:
    """Best-effort flash-attn build. Soft-fails to SDPA on any install error.

    flash-attn 2.7.3 imports ``psutil`` at build time but does not declare it as a
    build dependency. We install with ``--no-build-isolation`` (needs the venv's
    torch for CUDA), so ``psutil`` must already be in the TRELLIS venv.

    For uv project-managed builds elsewhere, the equivalent is::

        [tool.uv.extra-build-dependencies]
        flash-attn = ["psutil"]

    This bootstrap owns the Windows path; there is no lab pyproject for TRELLIS.
    """
    uv = shutil.which("uv")
    print("\nTrying flash-attn (optional; SDPA works if this fails)...", flush=True)
    # Build dep for flash-attn egg_info / setup under --no-build-isolation.
    psutil = subprocess.run(
        [uv, "pip", "install", "--python", str(py), "psutil"],
        check=False,
    )
    if psutil.returncode != 0:
        print("Could not install psutil (needed to build flash-attn); "
              "generate will use ATTN_BACKEND=sdpa.", flush=True)
        return
    done = subprocess.run(
        [uv, "pip", "install", "--python", str(py), "flash-attn==2.7.3",
         "--no-build-isolation"],
        env=extension_build_env(),
        check=False,
    )
    if done.returncode != 0:
        print("flash-attn not installed; generate will use ATTN_BACKEND=sdpa.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    args = parser.parse_args(argv)

    if host.host_platform() != host.NVIDIA:
        print("TRELLIS CUDA installs on Linux/Windows with an NVIDIA GPU. "
              "On Apple Silicon use scripts/bootstrap_trellis_space_macos.py. "
              "Nothing downloaded.")
        return 1

    print(announcement())
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to install without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if not host.find_nvcc():
        print("Warning: nvcc not found. Extension builds will likely fail. "
              "Install the CUDA toolkit (and on Windows, Visual Studio C++ build tools).",
              flush=True)

    ensure_clone()
    py = ensure_venv()
    install_torch(py)
    install_basic(py)
    install_extensions(py)
    try_flash_attn(py)
    print("\nDone. Pick TRELLIS.2 in Generate 3D. First run downloads ~14 GB of weights "
          "(DINOv3 is gated — run `hf auth login` first).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
