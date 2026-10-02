"""Hunyuan paint mesh_utils must not hard-import bpy at module load."""

from __future__ import annotations

from pathlib import Path

import patch_hunyuan_mesh_utils_blender as patch
import pytest

# Upstream top of mesh_utils + convert_obj_to_glb (anchors the patch cares about).
UPSTREAM = '''\
import os
import cv2
import bpy
import math
import numpy as np

def _setup_blender_scene():
    if "convert" not in bpy.data.scenes:
        bpy.data.scenes.new("convert")

''' + patch.CONVERT_OLD


def test_apply_removes_hard_bpy_import_and_routes_convert():
    out = patch.apply(UPSTREAM)
    assert patch.is_patched(out)
    assert "import bpy\n" not in out
    assert patch.MARKER in out
    assert "image_to_3dlab.obj_to_glb" in out
    assert "bpy.ops.wm.obj_import" not in out


def test_apply_is_idempotent():
    once = patch.apply(UPSTREAM)
    assert patch.apply(once) == once


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor not found"):
        patch.apply("import os\n")


def test_cli_patches_file(tmp_path):
    path = tmp_path / "mesh_utils.py"
    path.write_text(UPSTREAM, encoding="utf-8")
    assert patch.main(["--root", str(tmp_path), "--check"]) == 2
    assert patch.main(["--root", str(tmp_path)]) == 0
    assert patch.main(["--root", str(tmp_path), "--check"]) == 0
    assert "image_to_3dlab.obj_to_glb" in path.read_text(encoding="utf-8")


def test_hunyuan_generate_has_shape_only_and_defers_paint():
    source = Path(__file__).resolve().parents[1].joinpath(
        "scripts", "hunyuan_cuda_generate.py").read_text(encoding="utf-8")
    assert "--shape-only" in source
    # Import must sit after the shape-only early return (not the docstring mention).
    paint_import = "from textureGenPipeline import"
    assert paint_import in source
    assert "Hunyuan3DPaintPipeline" in source
    assert source.index("if args.shape_only:") < source.index(paint_import)
    assert "patch_hunyuan_mesh_utils_blender" in source
