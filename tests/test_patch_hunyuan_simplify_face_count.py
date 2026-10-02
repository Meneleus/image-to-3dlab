"""Hunyuan paint simplify must use face_count= for modern trimesh."""

from __future__ import annotations

from pathlib import Path

import patch_hunyuan_simplify_face_count as patch
import pytest

# Indentation must match the vendor anchor (8 spaces before courent = …).
UPSTREAM = (
    "def remesh_mesh(mesh_path, remesh_path, target_count=40000):\n"
    "    import trimesh\n"
    "    courent = trimesh.load(mesh_path, force=\"mesh\")\n"
    "        courent = courent.simplify_quadric_decimation(target_count)\n"
    "    courent.export(remesh_path)\n"
)


def test_apply_uses_face_count_keyword():
    out = patch.apply(UPSTREAM)
    assert patch.is_patched(out)
    assert patch.MARKER in out
    assert "face_count=target_count" in out
    assert "except TypeError:" in out
    # Fallback keeps the positional call at deeper indent; exact OLD line must go.
    assert patch.OLD not in out.splitlines(keepends=True)


def test_apply_is_idempotent():
    once = patch.apply(UPSTREAM)
    assert patch.apply(once) == once


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor not found"):
        patch.apply("def remesh_mesh():\n    pass\n")


def test_cli_patches_file(tmp_path):
    path = tmp_path / "simplify_mesh_utils.py"
    path.write_text(UPSTREAM, encoding="utf-8")
    assert patch.main(["--root", str(tmp_path), "--check"]) == 2
    assert patch.main(["--root", str(tmp_path)]) == 0
    assert patch.main(["--root", str(tmp_path), "--check"]) == 0
    text = path.read_text(encoding="utf-8")
    assert "face_count=target_count" in text
    assert patch.MARKER in text


def test_bootstrap_and_generate_wire_the_patch():
    repo = Path(__file__).resolve().parents[1]
    boot = (repo / "scripts" / "bootstrap_hunyuan_cuda.py").read_text(encoding="utf-8")
    gen = (repo / "scripts" / "hunyuan_cuda_generate.py").read_text(encoding="utf-8")
    assert "patch_hunyuan_simplify_face_count" in boot
    assert "patch_simplify_face_count" in boot
    assert "patch_hunyuan_simplify_face_count" in gen
    assert "_ensure_simplify_face_count_patch" in gen
