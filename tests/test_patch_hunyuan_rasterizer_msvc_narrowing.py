"""Hunyuan custom_rasterizer MSVC C2398 patch: real anchors, idempotent."""

from __future__ import annotations

from pathlib import Path

import patch_hunyuan_rasterizer_msvc_narrowing as patch
import pytest

# Minimal upstream-shaped excerpt covering both hierarchy helpers.
GRID_NEIGHBOR = """
    texture_positions[0] = torch::zeros({seq2pos.size() / 3, 3}, float_options);
    texture_positions[1] = torch::zeros({seq2pos.size() / 3}, float_options);
    grid_neighbors[i] = torch::zeros({grids[i].seq2grid.size(), 9}, int64_options);
    grid_evencorners[i] = torch::zeros({grids[i].seq2evencorner.size()}, int64_options);
    grid_oddcorners[i] = torch::zeros({grids[i].seq2oddcorner.size()}, int64_options);
    grid_downsamples[i] = torch::zeros({grids[i].downsample_seq.size()}, int64_options);

    texture_positions[0] = torch::zeros({seq2pos.size() / 3, 3}, float_options);
    texture_positions[1] = torch::zeros({seq2pos.size() / 3}, float_options);
    texture_feats[0] = torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options);
    grid_neighbors[i] = torch::zeros({grids[i].seq2grid.size(), 9}, int64_options);
    grid_evencorners[i] = torch::zeros({grids[i].seq2evencorner.size()}, int64_options);
    grid_oddcorners[i] = torch::zeros({grids[i].seq2oddcorner.size()}, int64_options);
    grid_downsamples[i] = torch::zeros({grids[i].downsample_seq.size()}, int64_options);
"""


def test_patch_source_casts_all_occurrences():
    old = "torch::zeros({seq2pos.size() / 3, 3}, float_options)"
    new = (
        "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3), "
        "static_cast<int64_t>(3)}, float_options)"
    )
    out, changed = patch.patch_source(GRID_NEIGHBOR, old, new)
    assert changed
    assert out.count(new) == 2
    assert old not in out
    out2, changed2 = patch.patch_source(out, old, new)
    assert changed2 is False and out2 == out


def test_longer_seq2pos_anchor_before_shorter():
    """``{size/3, 3}`` must be rewritten before ``{size/3}`` alone."""
    rels_olds = [(old, new) for _, old, new in patch.REPLACEMENTS]
    idx_2d = next(i for i, (o, _) in enumerate(rels_olds) if ", 3}" in o and "seq2pos" in o)
    idx_1d = next(
        i for i, (o, _) in enumerate(rels_olds)
        if "seq2pos.size() / 3}" in o and ", 3}" not in o
    )
    assert idx_2d < idx_1d


def test_patch_source_missing_anchor_raises():
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.patch_source("nope", "old", "new")


def test_apply_to_tree_patches_all_sites(tmp_path):
    root = tmp_path / "custom_rasterizer"
    path = root / patch.GRID_NEIGHBOR
    path.parent.mkdir(parents=True)
    path.write_text(GRID_NEIGHBOR, encoding="utf-8")

    states = patch.apply_to_tree(root)
    assert all(s.startswith("PATCHED") for s in states)
    states2 = patch.apply_to_tree(root)
    assert all(s.startswith("APPLIED") for s in states2)

    text = path.read_text(encoding="utf-8")
    assert "torch::zeros({seq2pos.size()" not in text
    assert "torch::zeros({grids[i]." not in text
    assert text.count("static_cast<int64_t>(seq2pos.size() / 3)") == 4  # 2x 2d + 2x 1d
    assert text.count("static_cast<int64_t>(grids[i].seq2grid.size())") == 2
    assert "static_cast<int64_t>(feat_channel)" in text
    # No unpatched size() brace-inits left for the reported patterns.
    assert "zeros({seq2feat.size()" not in text


def test_replacements_cover_user_reported_patterns():
    joined_old = "\n".join(old for _, old, _ in patch.REPLACEMENTS)
    for needle in (
        "seq2pos.size() / 3, 3",
        "seq2pos.size() / 3}",
        "seq2feat.size() / feat_channel",
        "seq2grid.size(), 9",
        "seq2evencorner.size()",
        "seq2oddcorner.size()",
        "downsample_seq.size()",
    ):
        assert needle in joined_old, needle


def test_against_upstream_checkout_if_present():
    candidates = [
        Path(__file__).resolve().parents[1]
        / "vendor" / "hunyuan3d-cuda" / "hy3dpaint" / "custom_rasterizer",
        Path("/tmp/i2l-hunyuan/hy3dpaint/custom_rasterizer"),
    ]
    root = next((p for p in candidates if p.is_dir()), None)
    if root is None:
        pytest.skip("no custom_rasterizer checkout present")
    for rel, old, new in patch.REPLACEMENTS:
        path = root / rel
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert new in text or old in text, f"{path}: neither old nor new present"


def test_bootstrap_calls_the_narrowing_patch():
    import bootstrap_hunyuan_cuda as boot
    source = Path(boot.__file__).read_text(encoding="utf-8")
    assert "patch_hunyuan_rasterizer_msvc_narrowing" in source
