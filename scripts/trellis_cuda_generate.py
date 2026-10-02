#!/usr/bin/env python3
"""Full image -> GLB generation through official TRELLIS.2 on NVIDIA CUDA.

Mirrors the CLI surface of ``trellis_space_generate.py`` so the viewer can drive Mac and
CUDA with the same settings. Uses ``microsoft/TRELLIS.2-4B``, stubs BiRefNet so BRIA
RMBG-2.0 is never constructed, and exports via ``o_voxel.postprocess.to_glb``.

    vendor/trellis2-cuda/.venv/bin/python scripts/trellis_cuda_generate.py \\
        input.png output.glb --resolution 1024 --seed 0

    ... trellis_cuda_generate.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.cuda_routes import TRELLIS_CUDA, venv_python

DEFAULT_VENDOR = TRELLIS_CUDA
DEMO_PARAMS: dict[str, Any] = {
    "resolution": "1024",
    "seed": 0,
    "decimation_target": 300_000,
    "texture_size": 2048,
    "sparse_structure": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7,
                         "rescale_t": 5.0},
    "shape_slat": {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5,
                   "rescale_t": 3.0},
    "tex_slat": {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0,
                 "rescale_t": 3.0},
    "remesh": {"remesh": True, "remesh_band": 1, "remesh_project": 0},
}
PIPELINE_TYPE_BY_RESOLUTION = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def pipeline_type_for_resolution(resolution: str) -> str:
    try:
        return PIPELINE_TYPE_BY_RESOLUTION[str(resolution)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported resolution {resolution!r}; expected one of "
            f"{sorted(PIPELINE_TYPE_BY_RESOLUTION)}"
        ) from exc


def configure_environment(vendor_root: Path, sparse_attn_backend: str) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Prefer flash_attn when installed; otherwise SDPA. Caller may force via flag.
    if sparse_attn_backend in ("flash_attn", "sdpa", "xformers"):
        os.environ["ATTN_BACKEND"] = sparse_attn_backend
        os.environ["SPARSE_ATTN_BACKEND"] = sparse_attn_backend
    else:
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        str(vendor_root / "cache" / "flex_gemm_autotune.json"),
    )


def verify_paths(vendor_root: Path) -> list[str]:
    problems = []
    checks = {
        ".venv python": venv_python(vendor_root),
        "trellis2 package": vendor_root / "trellis2",
        "o-voxel": vendor_root / "o-voxel",
    }
    for name, path in checks.items():
        if not path.exists():
            problems.append(f"missing {name}: {path}")
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment(vendor_root: Path, sparse_attn_backend: str) -> int:
    problems = verify_paths(vendor_root)
    if problems:
        for item in problems:
            print(f"  FAIL: {item}", flush=True)
        return 1
    configure_environment(vendor_root, sparse_attn_backend)
    sys.path.insert(0, str(vendor_root))
    import torch
    if not torch.cuda.is_available():
        print("  FAIL: torch.cuda.is_available() is False", flush=True)
        return 1
    try:
        import o_voxel  # noqa: F401
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: import: {exc}", flush=True)
        return 1
    print("  OK: CUDA TRELLIS.2 environment looks ready", flush=True)
    return 0


def pick_attn_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attn"
    except Exception:  # noqa: BLE001
        return "sdpa"


def generate(image: Path, output: Path, *, vendor_root: Path, resolution: str, seed: int,
             decimation_target: int, texture_size: int, sparse_attn_backend: str,
             allow_rembg: bool, keep_intermediate: bool) -> int:
    configure_environment(vendor_root, pick_attn_backend(sparse_attn_backend))
    sys.path.insert(0, str(vendor_root))

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg
    import o_voxel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    raw = Image.open(image)
    has_alpha = raw.mode == "RGBA" and raw.getextrema()[-1][0] < 255
    if not has_alpha and not allow_rembg:
        raise SystemExit(
            f"{image} has no transparent cut-out. Pass a pre-masked PNG, or "
            "--allow-rembg (uses rembg/u2net — never BRIA)."
        )

    print("[1/6] Loading TRELLIS.2 pipeline", flush=True)
    t0 = time.time()
    # Stub BiRefNet so the gated BRIA RMBG-2.0 constructor is never called.
    with patch.object(rembg, "BiRefNet", new=lambda **_k: None):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print(f"pipeline ready ({time.time() - t0:.0f}s)", flush=True)

    print("[2/6] Preprocessing image", flush=True)
    processed = pipeline.preprocess_image(raw)

    print("[3/6] Sampling (sparse + shape + material)", flush=True)
    sample_t0 = time.time()
    pipe_type = pipeline_type_for_resolution(resolution)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    meshes = pipeline.run(
        processed,
        seed=seed,
        pipeline_type=pipe_type,
        sparse_structure_sampler_params=dict(DEMO_PARAMS["sparse_structure"]),
        shape_slat_sampler_params=dict(DEMO_PARAMS["shape_slat"]),
        tex_slat_sampler_params=dict(DEMO_PARAMS["tex_slat"]),
    )
    mesh = meshes[0]
    print(f"sampling done ({time.time() - sample_t0:.0f}s)", flush=True)

    print("[4/6] Simplify for nvdiffrast", flush=True)
    if hasattr(mesh, "simplify"):
        mesh.simplify(16_777_216)

    print("[5/6] Bake + export GLB", flush=True)
    bake_t0 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=AABB,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output), extension_webp=True)
    print(f"wrote {output} ({time.time() - bake_t0:.0f}s)", flush=True)

    manifest = {
        "schema_version": 1,
        "generator": "trellis_cuda_generate.py",
        "port": "microsoft/TRELLIS.2 (CUDA)",
        "input": str(image),
        "output": str(output),
        "device": "cuda",
        "attn_backend": os.environ.get("ATTN_BACKEND"),
        "seed": seed,
        "pipeline_type": pipe_type,
        "params": {
            "resolution": resolution,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
        },
        "artifacts": {
            "glb": {"path": str(output), "sha256": sha256(output),
                    "bytes": output.stat().st_size},
        },
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[6/6] Manifest {manifest_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--resolution", default=DEMO_PARAMS["resolution"],
                        choices=sorted(PIPELINE_TYPE_BY_RESOLUTION))
    parser.add_argument("--seed", type=int, default=DEMO_PARAMS["seed"])
    parser.add_argument("--decimation-target", type=int,
                        default=DEMO_PARAMS["decimation_target"])
    parser.add_argument("--texture-size", type=int, default=DEMO_PARAMS["texture_size"])
    parser.add_argument("--sparse-attn-backend", default="auto",
                        choices=["auto", "flash_attn", "sdpa", "xformers", "mlx"])
    parser.add_argument("--allow-rembg", action="store_true")
    parser.add_argument("--no-keep-intermediate", dest="keep_intermediate",
                        action="store_false", default=True)
    args = parser.parse_args(argv)

    # mlx is Mac-only; ignore politely on CUDA.
    attn = "auto" if args.sparse_attn_backend == "mlx" else args.sparse_attn_backend

    if args.check:
        return check_environment(args.vendor_root, pick_attn_backend(attn))
    if args.image is None or args.output is None:
        parser.error("image and output are required unless --check")
    return generate(
        args.image, args.output, vendor_root=args.vendor_root,
        resolution=args.resolution, seed=args.seed,
        decimation_target=args.decimation_target, texture_size=args.texture_size,
        sparse_attn_backend=attn, allow_rembg=args.allow_rembg,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
