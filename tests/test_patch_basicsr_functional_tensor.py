"""basicsr must import rgb_to_grayscale from torchvision.transforms.functional."""

from __future__ import annotations

import patch_basicsr_functional_tensor as patch
import pytest

UPSTREAM = """\
import cv2
import torch
from torchvision.transforms.functional_tensor import rgb_to_grayscale

def sigma_matrix2(sig_x, sig_y, theta):
    return None
"""


def test_apply_rewrites_functional_tensor_import():
    out = patch.apply(UPSTREAM)
    assert patch.is_patched(out)
    assert patch.OLD not in out
    assert "torchvision.transforms.functional import rgb_to_grayscale" in out
    assert patch.MARKER in out


def test_apply_is_idempotent():
    once = patch.apply(UPSTREAM)
    assert patch.apply(once) == once


def test_already_fixed_upstream_is_noop():
    fixed = UPSTREAM.replace(patch.OLD, patch.NEW)
    assert patch.apply(fixed) == fixed
    assert patch.is_patched(fixed)


def test_missing_anchor_fails_loudly():
    with pytest.raises(SystemExit, match="anchor not found"):
        patch.apply("import torch\n")


def test_cli_patches_path(tmp_path):
    path = tmp_path / "degradations.py"
    path.write_text(UPSTREAM, encoding="utf-8")
    assert patch.main(["--path", str(path), "--check"]) == 2
    assert patch.main(["--path", str(path)]) == 0
    assert patch.main(["--path", str(path), "--check"]) == 0
    text = path.read_text(encoding="utf-8")
    assert patch.OLD not in text
    assert patch.NEW in text


def test_bootstrap_patches_basicsr_after_realesrgan(monkeypatch, tmp_path):
    import bootstrap_hunyuan_cuda as boot

    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    monkeypatch.setattr(boot.host, "driver_cuda_version", lambda: "13.0")
    monkeypatch.setattr(
        boot.host, "torch_cuda_index",
        lambda *_: "https://download.pytorch.org/whl/cu130",
    )
    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.trellis_boot, "patch_torch_cpp_extension_cxx20", lambda *_: None)
    monkeypatch.setattr(boot, "patch_custom_rasterizer_msvc_narrowing", lambda *_: None)
    monkeypatch.setattr(boot, "patch_mesh_utils_blender", lambda *_: None)
    monkeypatch.setattr(boot, "install_cuda_extension", lambda *a, **k: None)
    monkeypatch.setattr(boot, "VENDOR", tmp_path)
    monkeypatch.setattr(boot.urllib.request, "urlretrieve", lambda *a, **k: None)
    seen: list[list[str]] = []
    monkeypatch.setattr(boot, "run", lambda cmd, **kw: seen.append(list(cmd)))
    vendor_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    boot.install_code(vendor_py)
    assert any("patch_basicsr_functional_tensor.py" in str(c) for c in seen)
    assert any("from realesrgan import RealESRGANer" in str(c) for c in seen)
