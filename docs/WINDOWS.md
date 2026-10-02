# Windows + NVIDIA

How to run this lab on a Windows PC with an NVIDIA graphics card.

Mac and Linux docs stay in the [README](../README.md). This page is only the Windows
route.

## What works here

| Route | In this lab on Windows? | Notes |
|---|---|---|
| **Pixal3D** | Yes | Downloads a CUDA 12 prebuilt (driver **575+**). No Visual Studio needed. |
| **Generate Image** (Qwen-Image) | Yes | Prebuilt CUDA `sd-cli`; CUDA runtime is bundled. |
| **Stable Fast 3D** | Yes | Needs **Visual Studio C++ tools** (`cl` on PATH) to compile the texture baker. Bootstrap patches MSVC/CUDA 13 flags first. |
| **TRELLIS.2** | Yes (CUDA) | `scripts/bootstrap_trellis_cuda.py` clones official microsoft/TRELLIS.2 into `vendor/trellis2-cuda`. Needs CUDA toolkit + VS C++ tools. Microsoft only documents Linux; Windows is best-effort. |
| **Hunyuan3D-2.1** | Yes (CUDA) | `scripts/bootstrap_hunyuan_cuda.py` clones official Tencent Hunyuan3D-2.1 into `vendor/hunyuan3d-cuda`. Officially supports Windows. |
| **Hunyuan3D-MLX (dgrauet)** | No | Apple MLX only. Prefer the CUDA Hunyuan route above. |
| **Finish** (retopo, Pixel Match, compress) | Yes | Needs **Blender 4.2+**. Repaint of sides/back is Mac-only (MLX). |

Shell scripts named `*_macos.sh` are Apple Silicon only. On Windows use the Python
bootstraps (`scripts/bootstrap_*.py`), the PowerShell wrappers, or **Setup & Status**.

## Prerequisites

1. **64-bit Windows 10 or 11**
2. **NVIDIA GPU** with a current driver: `nvidia-smi` must list the card.
   - PyTorch CUDA wheels need driver support for **CUDA 12.8+**
   - Pixal3D’s prebuilt needs **CUDA 12.9+** (driver **575** or newer)
3. **git** — [git-scm.com/download/win](https://git-scm.com/download/win)
4. **Blender 4.2+** (for Finish) — [blender.org/download](https://www.blender.org/download/)
5. **For SF3D, TRELLIS.2, or Hunyuan3D:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the **Desktop development with C++** workload, plus a **CUDA Toolkit** whose
   major version matches the PyTorch build (`python -m image_to_3dlab.host torch-index`).
6. **Not needed:** Homebrew, Xcode, Metal, or MLX.

## Install (one line)

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.ps1 | iex
```

That clones (or updates) into `%USERPROFILE%\image-to-3dlab`, installs Python 3.11 via
`uv`, and installs **PyTorch with CUDA** on an NVIDIA machine. It downloads **no model
weights**.

Options:

```powershell
$env:I3D_DIR = "D:\lab"          # install folder
$env:I3D_REF = "main"            # branch or tag
$env:I3D_REPO = "https://github.com/YOU/image-to-3dlab.git"  # a fork
$env:I3D_YES = "1"               # no prompts (scripts / agents)
irm https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.ps1 | iex
```

### From a fresh clone (no one-line installer)

```powershell
git clone https://github.com/Bingeljell/image-to-3dlab.git
cd image-to-3dlab
uv venv --python 3.11 .venv
.venv\Scripts\python.exe -m image_to_3dlab.host torch-index
# If that printed a URL (e.g. https://download.pytorch.org/whl/cu128), install CUDA torch:
uv pip install --python .venv\Scripts\python.exe --index-url <that-url> torch torchvision
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Start the viewer

```powershell
cd $env:USERPROFILE\image-to-3dlab   # or your I3D_DIR
.venv\Scripts\python.exe viewer\serve.py
```

Open the URL it prints. Go to **Setup & Status**, pick a backend, read the size and
licence, then confirm. Weights only download after you say yes.

## Setting up TRELLIS.2 and Hunyuan3D in the lab

These are first-class Generate 3D backends on Windows/Linux NVIDIA — same dropdown and
CLI pattern as on Mac, different vendor tree under the hood.

### TRELLIS.2 (CUDA)

```powershell
.venv\Scripts\python.exe scripts\bootstrap_trellis_cuda.py
# or: .\scripts\bootstrap_trellis_cuda.ps1
```

What it does: clones `microsoft/TRELLIS.2` → `vendor/trellis2-cuda/`, installs CUDA
PyTorch + deps, compiles nvdiffrast / cumesh / flexgemm / o-voxel. **No weights yet.**

On Windows the bootstrap **sets the MSVC/CUDA compile flags for you** — you should not
need to hand-set `CXXFLAGS` / `NVCC_FLAGS`. See [CUDA extension builds](#cuda-extension-builds-trellis2)
below if a compile still fails.

Then in the viewer pick **TRELLIS.2** → Generate. First run downloads ~14 GB
(`microsoft/TRELLIS.2-4B`). Before that: request gated DINOv3 access and run
`hf auth login`.

CLI:

```powershell
vendor\trellis2-cuda\.venv\Scripts\python.exe scripts\trellis_cuda_generate.py `
  input.png output\out.glb --resolution 1024 --seed 0
```

### Hunyuan3D-2.1 (CUDA)

```powershell
.venv\Scripts\python.exe scripts\bootstrap_hunyuan_cuda.py
# or: .\scripts\bootstrap_hunyuan_cuda.ps1
```

What it does: clones `Tencent-Hunyuan/Hunyuan3D-2.1` → `vendor/hunyuan3d-cuda/`, builds
the paint rasterizer (`custom_rasterizer`), fetches ~10 GB weights (asks first unless
`--yes`). Runtime packages go into **`vendor/hunyuan3d-cuda/.venv`**, not the lab root
`.venv` — Generate 3D uses that interpreter. That includes **pymeshlab** (`hy3dshape`
postprocess), **rembg**, and **`onnxruntime-gpu`** on Windows (plain `onnxruntime` on
Linux) so rembg can import ORT. On Windows it also installs `msvc-runtime` for
pymeshlab’s DLLs, sets `DISTUTILS_USE_SDK=1` and the same C++20 / `/Zc:preprocessor`
flags as TRELLIS, and patches the rasterizer for MSVC (`size_t`→`int64_t` shapes,
`long`→`int64_t` LibTorch `data_ptr`).

**Paint vs Finish’s Blender:** upstream paint does `import bpy` in
`DifferentiableRenderer/mesh_utils.py` for OBJ→GLB. This lab does **not** pip-install
`bpy` into the Hunyuan venv (it fights Windows / Finish’s `blender.exe`). Bootstrap
runs `patch_hunyuan_mesh_utils_blender.py` so conversion uses **`I2L_BLENDER` /
`find_blender()`** (same Blender Finish uses), then optional pip `bpy`, then trimesh.
Shape-only: `… hunyuan_cuda_generate.py … --shape-only` skips paint entirely.

**Landmine:** DifferentiableRenderer has **no Windows `setup.py`** — only
`compile_mesh_painter.sh` (Linux). `mesh_inpaint_processor` may be missing; paint can
still run but vertex inpaint may warn / degrade.

**Shell:** CUDA extension builds need the MSVC + CUDA toolchains on PATH. A plain
PowerShell often has neither. Before bootstrap (or any local/agent automation that
compiles), run something that loads the **64-bit** VS developer environment, e.g.:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
  -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
  -property installationPath
# then, from that install root:
#   .\Common7\Tools\VsDevCmd.bat -arch=amd64
# or open "x64 Native Tools Command Prompt for VS".
```

`VsDevCmd.bat` without `-arch=amd64` can leave you on x86 `cl`, which will not link
these CUDA extensions.

**Licence:** Tencent Community License — not for EU / UK / South Korea.

CLI:

```powershell
vendor\hunyuan3d-cuda\.venv\Scripts\python.exe scripts\hunyuan_cuda_generate.py `
  input.png output\out.glb --model 2.1
```

In the viewer this is the **Hunyuan3D (shape + paint)** dropdown entry (same id as the
Mac MLX Xiong route; the lab picks CUDA automatically on NVIDIA).

### Stable Fast 3D (CUDA baker)

```powershell
.venv\Scripts\python.exe scripts\bootstrap_sf3d.py
# or: .\scripts\bootstrap_sf3d.ps1
```

Compiles `texture_baker` (CUDA when `nvcc` matches PyTorch) and `uv_unwrapper`. On
Windows, `scripts/bootstrap_sf3d.py` runs `patch_sf3d_windows_cuda_ext.py` first so
those `setup.py` files get MSVC-safe cxx flags and nvcc host flags
(`/std:c++20`, `/Zc:preprocessor` via `-Xcompiler=`) — upstream only adds
`/Zc:preprocessor` inside `debug_mode`, which is why a normal release build hits
CCCL **C1189** on CUDA 13 + modern MSVC. The install env also gets the same
`DISTUTILS_USE_SDK=1` / `CL` Zc flags as TRELLIS/Hunyuan.

**Shell:** same as TRELLIS/Hunyuan — `VsDevCmd.bat -arch=amd64` (or x64 Native Tools)
before bootstrap so `cl` is the 64-bit toolchain.

If compile fails and `cl` is already on PATH, the hint names the preprocessor/flags
issue — it does **not** tell you to install Build Tools again.

## CLI (quick routes)

```powershell
# Pixal3D
.venv\Scripts\python.exe scripts\bootstrap_pixal3d.py
.venv\Scripts\python.exe scripts\pixal3d_generate.py input.png output.glb --seed 42

# Generate Image
.venv\Scripts\python.exe scripts\bootstrap_qwen_image.py

# Stable Fast 3D
.venv\Scripts\python.exe scripts\bootstrap_sf3d.py

# Finish (needs Blender)
.venv\Scripts\python.exe scripts\retopo_repaint.py generated.glb source.png finished.glb --faces 40000 --skip-paint
```

If Blender is installed somewhere unusual:

```powershell
$env:I2L_BLENDER = "C:\Path\To\blender.exe"
```

## CUDA extension builds (TRELLIS.2)

TRELLIS.2 compiles several CUDA packages (**nvdiffrast, nvdiffrec, CuMesh, FlexGEMM,
o-voxel** — every `setup.py` the bootstrap installs). On Windows that needs:

1. **CUDA Toolkit 12.8+** (13.x is fine) with `nvcc` on PATH or under the usual
   `NVIDIA GPU Computing Toolkit\CUDA\v*` folder.
2. **Visual Studio 2022** (17) or **Visual Studio 18** with the **Desktop development
   with C++** workload (MSVC 14.3x / 14.5x). Open the **x64 Native Tools** prompt, or
   run `VsDevCmd.bat -arch=amd64` so `cl` is the 64-bit toolchain, before bootstrapping.
   A plain Shell / agent session usually has no `cl` on PATH.
3. A driver new enough for that toolkit (and for your GPU — **Blackwell / sm_120**
   needs a recent driver plus a toolkit that knows `sm_120`).

`scripts/bootstrap_trellis_cuda.py` forces **only C++20** on the compile line:

| Step | What it does |
|---|---|
| Env | `DISTUTILS_USE_SDK=1`, `CL` = `/Zc:preprocessor /Zc:__cplusplus` only (no `/std:` in env; no bare MSVC flags in `CXXFLAGS` — those leak into nvcc) |
| `setup.py` rewrite | Every extension: `c++17` → `c++20`; inject MSVC CCCL flags into known shapes (incl. nvdiffrast’s warning-only Windows list) |
| Runtime hook | **Same hook for every package**: appends `/std:c++20` + `/Zc:preprocessor` to `cxx`/`nvcc`, sanitizes ninja/spawn so 17 and 20 never mix |
| Torch patch | Rewrites `c++17` → `c++20` inside this venv’s `torch/utils/cpp_extension.py` (older wheels hardcode `-std=c++17` into ninja) |
| o-voxel source | `scripts/patch_ovoxel_msvc_narrowing.py`: `size_t`→`int64_t` shapes (C2398), strip upstream `1e-6d`/`0.0d` (C3688), `size_t`→`int` in `int4` (C4838) |
| Clean | Deletes `build/`, `*.egg-info`, `build.ninja` before `pip install` |
| flash-attn (optional) | Installs `psutil` first, then `flash-attn==2.7.3` with `--no-build-isolation`. Soft-fails to SDPA if the CUDA build fails. (uv projects elsewhere: `[tool.uv.extra-build-dependencies] flash-attn = ["psutil"]`) |

The post-patch check requires the CCCL **ensure** hook (`_i2l_ensure_msvc_cccl_flags`), not
merely the string `/std:c++20` somewhere in the file — an older check mistook hook
source literals for real compile flags and aborted nvdiffrast.

**Gotcha (cl D9025):** several places inject `/std:` — FlexGEMM `setup.py`, older PyTorch
ninja rules, and `CL` under nvcc `--use-local-env`. If **both** `/std:c++20` and
`/std:c++17` appear, MSVC warns `overriding …` back and forth and the final standard
is wrong. Do **not** hand-set `CL=/std:c++20` while debugging; let the bootstrap own it.
Clear stale builds: delete `vendor\trellis2-cuda\.i2l-build\FlexGEMM` (or at least its
`build\` folder / `build.ninja`) and re-run.

**Blackwell (sm_120):** if nvcc builds for the wrong arch, set
`TORCH_CUDA_ARCH_LIST=12.0` in the same shell before re-running the bootstrap.

## Common snags

| Symptom | Fix |
|---|---|
| Installer says no NVIDIA GPU | Install/update the NVIDIA driver; reopen PowerShell; `nvidia-smi` must work. |
| PyTorch stays on CPU | Re-run the installer, or install from the URL `python -m image_to_3dlab.host torch-index` prints. |
| Pixal3D refuses the prebuilt | Update the driver to **575+**. |
| TRELLIS/Hunyuan/SF3D compile fails and `cl` is missing | Install VS / Build Tools (C++) + a CUDA toolkit matching PyTorch; open “x64 Native Tools” / `VsDevCmd.bat -arch=amd64` and re-run. |
| SF3D `texture_baker` C1189 / traditional preprocessor / Linux flags on MSVC | Pull latest; bootstrap runs `patch_sf3d_windows_cuda_ext.py`. Wipe `vendor\stable-fast-3d\texture_baker\build` (and `uv_unwrapper\build`) and re-run from an x64 Native Tools shell. |
| `D9025` flipping `/std:c++20` ↔ `/std:c++17` on FlexGEMM | Torch and/or `setup.py` still emit C++17, or a stale `build\` ninja file. Pull latest, delete `vendor\trellis2-cuda\.i2l-build\FlexGEMM`, unset any hand-set `CL`/`CXXFLAGS` `/std:`, re-run bootstrap. See [above](#cuda-extension-builds-trellis2). |
| Bootstrap says nvdiffrast is missing `/Zc:preprocessor` after “Patched…” | Fixed: pull latest (CCCL ensure hook). Delete `vendor\trellis2-cuda\.i2l-build\nvdiffrast` and re-run. |
| o-voxel MSVC `C2398` / `C3688` (`…d` float suffix) / `C4838` | Pull latest; bootstrap runs `patch_ovoxel_msvc_narrowing.py`. Wipe `vendor\trellis2-cuda\o-voxel\build` and re-run. |
| flash-attn fails with `No module named 'psutil'` | Fixed: bootstrap installs `psutil` into the TRELLIS venv before the optional flash-attn build (`--no-build-isolation`). Still soft-fails to SDPA if the CUDA extension itself will not compile. Re-run bootstrap (or just let it retry the flash-attn step). |
| Hunyuan `custom_rasterizer`: VC env active but `DISTUTILS_USE_SDK` not set | Fixed: Hunyuan bootstrap uses the same Windows CUDA build env as TRELLIS (`DISTUTILS_USE_SDK=1`, `/Zc:preprocessor`). Pull latest and re-run from an x64 Native Tools shell. |
| Hunyuan / TRELLIS: `nvcc fatal: A single input file is required` with `/Zc:` or `/std:c++20` on the nvcc line | Bare MSVC host flags reached nvcc. Pull latest (nvcc sanitizer wraps them as `-Xcompiler=…`; env no longer puts them in `CXXFLAGS`). Wipe the extension `build\` folder and re-run. |
| Hunyuan `custom_rasterizer` MSVC `C2398` / `LNK2001` `data_ptr<long>` | Pull latest; bootstrap runs `patch_hunyuan_rasterizer_msvc_narrowing.py` (`size_t`→`int64_t` shapes, `long`→`int64_t` for LibTorch). Wipe `hy3dpaint\custom_rasterizer\build` and re-run from `VsDevCmd.bat -arch=amd64`. |
| Hunyuan `No module named 'pymeshlab'` / `'onnxruntime'` / `'realesrgan'` / `'pytorch_lightning'` / `'fast_simplification'` | Pull latest; re-run `scripts\bootstrap_hunyuan_cuda.py --yes --code-only`. It installs those into `vendor\hunyuan3d-cuda\.venv` — the interpreter Generate 3D uses, not the lab root `.venv`. |
| Hunyuan `functional_tensor` / `rgb_to_grayscale` after installing realesrgan | basicsr (pulled by realesrgan 0.3.0) still imports removed `torchvision.transforms.functional_tensor`. Pull latest; bootstrap runs `patch_basicsr_functional_tensor.py` then checks `from realesrgan import RealESRGANer`. Or: `python scripts\patch_basicsr_functional_tensor.py --python vendor\hunyuan3d-cuda\.venv\Scripts\python.exe`. |
| Hunyuan paint `No module named 'bpy'` | Do **not** need pip `bpy`. Pull latest + re-run Hunyuan `--code-only` (patches `mesh_utils`). Install Blender 4.2+ (Finish) or set `I2L_BLENDER`. Or run with `--shape-only` to skip paint. |
| Hunyuan paint `ValueError: target_reduction must be between 0 and 1` | Newer trimesh treats the first arg of `simplify_quadric_decimation` as percent (0–1), not face count. Pull latest; bootstrap / generate run `patch_hunyuan_simplify_face_count.py` (`face_count=`). Or: `python scripts\patch_hunyuan_simplify_face_count.py`. |
| HF hub `WinError 1314` / symlink privilege on Windows | Hugging Face cache wants symlinks; without Developer Mode that fails. Pull latest — Generate / Setup / Hunyuan CLI set `HF_HUB_DISABLE_SYMLINKS=1` (copies into `snapshots/` instead). Or enable Windows Developer Mode. |
| rembg / Pixal `Conv_0` / cuDNN unavailable | rembg starts ONNX Runtime before torch, so `cudnn64_9.dll` in `.venv\Lib\site-packages\torch\lib` is missed (CUDA Toolkit 13.2 does not ship cuDNN 9). Pull latest: `matte.prepare_onnxruntime_cuda()` prepends that folder for **this process only** (`add_dll_directory` + PATH — not User PATH) and calls `onnxruntime.preload_dlls()` when present. Optional extra: `onnxruntime-gpu[cudnn]`. |
| `cl` / `nvcc` not found in a plain Shell / agent session | Load the VS env first: `VsDevCmd.bat -arch=amd64` (or x64 Native Tools). Plain PowerShell has no `cl` on PATH. |
| TRELLIS fails only on Windows | Microsoft tests Linux; try the same bootstrap on Linux NVIDIA, or WSL2 with GPU. |
| Generate Image is very slow / CPU | `nvidia-smi` must see the card; update the driver and reopen the terminal. |
| Finish cannot find Blender | Install from blender.org, or set `I2L_BLENDER`. |

## What we do not do for you

- We never download model weights until you confirm in Setup & Status (or pass `--yes`).
- We never install GPU drivers, the CUDA Toolkit, Visual Studio, or Blender.

Windows support is real but still lightly tested. If something breaks, please say so in
Discussions or an issue — include your GPU, driver version (`nvidia-smi`), and the exact
command.
