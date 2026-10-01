"""Pure helpers for the CUDA TRELLIS generate wrapper."""

from __future__ import annotations

import trellis_cuda_generate as gen
import pytest


def test_pipeline_type_maps_demo_resolutions():
    assert gen.pipeline_type_for_resolution("512") == "512"
    assert gen.pipeline_type_for_resolution("1024") == "1024_cascade"
    assert gen.pipeline_type_for_resolution("1536") == "1536_cascade"
    with pytest.raises(ValueError, match="unsupported"):
        gen.pipeline_type_for_resolution("2048")


def test_pick_attn_honours_an_explicit_backend():
    assert gen.pick_attn_backend("xformers") == "xformers"
    assert gen.pick_attn_backend("sdpa") == "sdpa"


def test_pick_attn_auto_without_flash_uses_sdpa(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_attn" or (fromlist and "flash_attn" in name):
            raise ImportError("nope")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gen.pick_attn_backend("auto") == "sdpa"
