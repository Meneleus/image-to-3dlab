"""SF3D texture_baker / uv_unwrapper must compile on Windows MSVC + CUDA 13."""

from __future__ import annotations

import patch_sf3d_windows_cuda_ext as patch
import pytest

# Exact upstream blocks (nested Windows append under debug_mode for texture_baker).
TEXTURE_BAKER_SETUP = '''\
import platform

def get_extensions():
    debug_mode = False
    use_native_arch = True
    extra_link_args = []
''' + patch.TEXTURE_BAKER_OLD + '''
    return extra_compile_args
'''

UV_UNWRAPPER_SETUP = '''\
def get_extensions():
    debug_mode = False
    is_mac = False
    use_native_arch = True
    extra_link_args = []
''' + patch.UV_UNWRAPPER_OLD + '''
    return extra_compile_args
'''


def test_texture_baker_patch_injects_zc_preprocessor_and_drops_gcc_flags():
    out, changed = patch.patch_source(
        TEXTURE_BAKER_SETUP, patch.TEXTURE_BAKER_OLD, patch.TEXTURE_BAKER_NEW)
    assert changed
    assert patch.MARKER in out
    assert "/Zc:preprocessor" in out
    assert "-Xcompiler=/Zc:preprocessor" in out
    assert "/std:c++20" in out
    assert "-fopenmp" in out  # still on the non-Windows branch
    # Release Windows path must not keep gcc-only cxx flags as the only list.
    assert 'if platform.system() == "Windows":' in out
    assert '"/openmp"' in out or "'/openmp'" in out


def test_uv_unwrapper_patch_uses_msvc_cxx_flags():
    out, changed = patch.patch_source(
        UV_UNWRAPPER_SETUP, patch.UV_UNWRAPPER_OLD, patch.UV_UNWRAPPER_NEW)
    assert changed
    assert patch.MARKER in out
    assert "/Zc:preprocessor" in out
    assert "/std:c++20" in out
    assert "-march=native" in out  # remains on the else branch


def test_apply_is_idempotent(tmp_path):
    root = tmp_path
    (root / "texture_baker").mkdir()
    (root / "uv_unwrapper").mkdir()
    (root / "texture_baker" / "setup.py").write_text(TEXTURE_BAKER_SETUP)
    (root / "uv_unwrapper" / "setup.py").write_text(UV_UNWRAPPER_SETUP)

    first = patch.apply_to_tree(root)
    assert all(s.startswith("PATCHED") for s in first)
    second = patch.apply_to_tree(root)
    assert all(s.startswith("APPLIED") for s in second)
    assert all(s.startswith("APPLIED") for s in patch.check_tree(root))


def test_missing_anchor_fails_loudly():
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.patch_source("nope", patch.TEXTURE_BAKER_OLD, patch.TEXTURE_BAKER_NEW)


def test_cli_check_and_apply(tmp_path):
    root = tmp_path
    (root / "texture_baker").mkdir()
    (root / "uv_unwrapper").mkdir()
    (root / "texture_baker" / "setup.py").write_text(TEXTURE_BAKER_SETUP)
    (root / "uv_unwrapper" / "setup.py").write_text(UV_UNWRAPPER_SETUP)
    assert patch.main(["--root", str(root), "--check"]) == 2
    assert patch.main(["--root", str(root)]) == 0
    assert patch.main(["--root", str(root), "--check"]) == 0
