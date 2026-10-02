#!/usr/bin/env python3
"""End-to-end Hunyuan3D-2.1 (official CUDA) generation: image -> textured GLB.

CLI surface matches ``hunyuan_mlx_xiong_generate.py`` closely enough for the viewer
(`--model`, octree/seed/decimation/paint_*). On CUDA, `--model` selects the HF subfolder
under the local weights cache (default 2.1, the official CUDA release).

    vendor/hunyuan3d-cuda/.venv/bin/python scripts/hunyuan_cuda_generate.py \\
        input.png output.glb --model 2.1
    vendor/hunyuan3d-cuda/.venv/bin/python scripts/hunyuan_cuda_generate.py \\
        input.png output.glb --shape-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.cuda_routes import HUNYUAN_CUDA
from image_to_3dlab.hf_hub_env import ensure_process_windows_hf_hub_env
from image_to_3dlab.matte import prepare_onnxruntime_cuda

VENDOR = HUNYUAN_CUDA
# Local layout written by bootstrap_hunyuan_cuda.py --weights
SHAPE_SUBDIRS = {
    "2.1": "hunyuan3d-dit-v2-1",
    "2.0": "hunyuan3d-dit-v2-0",
    "2.0-turbo": "hunyuan3d-dit-v2-0-turbo",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=sorted(SHAPE_SUBDIRS), default="2.1")
    parser.add_argument("--octree-resolution", type=int, default=512,
                        choices=[256, 384, 512, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="ignored on CUDA (kept for CLI parity with the MLX wrapper)")
    parser.add_argument("--compile-dit", action="store_true", default=False)
    parser.add_argument("--no-compile-dit", dest="compile_dit", action="store_false")
    parser.add_argument("--octree-decode", action="store_true", default=True)
    parser.add_argument("--no-octree-decode", dest="octree_decode", action="store_false")
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    parser.add_argument(
        "--shape-only",
        action="store_true",
        help="Stop after the shape GLB; skip paint (no textureGenPipeline / Blender OBJ→GLB).",
    )
    return parser.parse_args(argv)


def _prepare_paths() -> None:
    for sub in ("hy3dshape", "hy3dpaint"):
        path = VENDOR / sub
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def _ensure_mesh_utils_patch() -> None:
    """Idempotent: drop hard ``import bpy`` even if bootstrap was not re-run."""
    diff = VENDOR / "hy3dpaint" / "DifferentiableRenderer"
    script = REPO / "scripts" / "patch_hunyuan_mesh_utils_blender.py"
    if not (diff / "mesh_utils.py").is_file() or not script.is_file():
        return
    subprocess.run([sys.executable, str(script), "--root", str(diff)], check=False)


def shape_model_path(model: str) -> str:
    """HF id or local weights directory the shape pipeline can load."""
    local = VENDOR / "weights" / SHAPE_SUBDIRS[model]
    if local.is_dir():
        return str(VENDOR / "weights")
    # Fall back to the Hugging Face repo (downloads on first use if cache is empty).
    return "tencent/Hunyuan3D-2.1" if model == "2.1" else "tencent/Hunyuan3D-2"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Paint may hf_hub_download into ~/.cache/huggingface; WinError 1314 without this.
    ensure_process_windows_hf_hub_env()
    if not (VENDOR / "hy3dshape").is_dir():
        raise SystemExit(
            f"Hunyuan CUDA checkout missing at {VENDOR}. "
            "Run: python scripts/bootstrap_hunyuan_cuda.py"
        )
    _prepare_paths()
    if not args.shape_only:
        _ensure_mesh_utils_patch()
    t0 = time.time()

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception:  # noqa: BLE001, S110 — optional vendor shim
        pass

    # rembg/ORT CUDA EP needs cudnn64_9.dll from torch\\lib on Windows — before ORT loads.
    prepare_onnxruntime_cuda()

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover
    from PIL import Image

    image = Image.open(args.image).convert("RGBA")
    if image.getextrema()[-1][0] >= 255:
        print(f"cutting out background ({time.time() - t0:.0f}s)", flush=True)
        image = BackgroundRemover()(image)

    model_path = shape_model_path(args.model)
    print(f"loading shape pipeline (model={args.model})... ({time.time() - t0:.0f}s)",
          flush=True)
    kwargs = {}
    subfolder = SHAPE_SUBDIRS[args.model]
    if model_path.startswith("tencent/"):
        kwargs["subfolder"] = subfolder
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, **kwargs)
    print(f"shape pipeline ready ({time.time() - t0:.0f}s)", flush=True)

    mesh = shape(
        image=image,
        num_inference_steps=30,
        octree_resolution=args.octree_resolution,
        generator=__import__("torch").Generator("cuda").manual_seed(args.seed),
    )[0]
    print(f"shape generated ({time.time() - t0:.0f}s)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_mesh = args.output.with_name(args.output.stem + "_shape.glb")
    mesh.export(str(raw_mesh))

    if hasattr(mesh, "faces") and len(getattr(mesh, "faces", [])) > args.decimation_target:
        try:
            import fast_simplification
            import numpy as np
            import trimesh

            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.faces)
            v_out, f_out = fast_simplification.simplify(
                v, f, target_count=args.decimation_target)
            trimesh.Trimesh(vertices=v_out, faces=f_out, process=False).export(str(raw_mesh))
            print(f"simplified to {len(f_out):,} faces ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"remesh skipped ({exc})", flush=True)

    if args.shape_only:
        shutil.copy2(raw_mesh, args.output)
        print(f"shape-only done ({time.time() - t0:.0f}s): {args.output}", flush=True)
        manifest = {
            "schema_version": 1,
            "generator": "hunyuan_cuda_generate.py",
            "port": "Tencent-Hunyuan/Hunyuan3D-2.1 (CUDA)",
            "backend": "hunyuan-cuda",
            "input": str(args.image),
            "output": str(args.output),
            "model": args.model,
            "seed": args.seed,
            "octree_resolution": args.octree_resolution,
            "decimation_target": args.decimation_target,
            "paint": None,
            "shape_only": True,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        return 0

    # Deferred so --shape-only never imports paint / mesh_utils / bpy.
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    print(f"paint diffusion starting ({time.time() - t0:.0f}s)", flush=True)
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=args.paint_res)
    conf.realesrgan_ckpt_path = str(VENDOR / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = str(VENDOR / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = str(VENDOR / "hy3dpaint" / "hunyuanpaintpbr")
    paint = Hunyuan3DPaintPipeline(conf)
    painted = paint(
        mesh_path=str(raw_mesh),
        image_path=str(args.image),
        output_mesh_path=str(args.output),
    )
    print(f"paint finished ({time.time() - t0:.0f}s): {painted}", flush=True)

    manifest = {
        "schema_version": 1,
        "generator": "hunyuan_cuda_generate.py",
        "port": "Tencent-Hunyuan/Hunyuan3D-2.1 (CUDA)",
        "backend": "hunyuan-cuda",
        "input": str(args.image),
        "output": str(args.output),
        "model": args.model,
        "seed": args.seed,
        "octree_resolution": args.octree_resolution,
        "decimation_target": args.decimation_target,
        "paint": {
            "seed": args.paint_seed,
            "res": args.paint_res,
            "steps": args.paint_steps,
            "tex": args.paint_tex,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
