"""MSVC narrowing patch for o-voxel: real anchors, idempotent, covers reported sites."""

from __future__ import annotations

from pathlib import Path

import pytest

import patch_ovoxel_msvc_narrowing as patch


FILTER_NEIGHBOR = '''
    // Pack the deltas into a uint8 tensor
    torch::Tensor delta = torch::zeros({N, C}, torch::dtype(torch::kUInt8));
    // Pack the attribute into a uint8 tensor
    torch::Tensor attr = torch::zeros({N, C}, torch::dtype(torch::kUInt8));
'''

FILTER_PARENT = '''
    torch::Tensor delta = torch::zeros({N_leaf, C}, torch::kUInt8);
    torch::Tensor attr = torch::zeros({N_leaf, C}, torch::kUInt8);
'''

SVO = '''
    torch::Tensor svo_tensor = torch::from_blob(svo.data(), {svo.size()}, torch::kUInt8).clone();
    torch::Tensor codes_tensor = torch::from_blob(codes.data(), {codes.size()}, torch::kInt32).clone();
'''


def test_patch_source_casts_size_t_shapes():
    old = "torch::Tensor delta = torch::zeros({N, C}, torch::dtype(torch::kUInt8));"
    new = (
        "torch::Tensor delta = torch::zeros({static_cast<int64_t>(N), "
        "static_cast<int64_t>(C)}, torch::dtype(torch::kUInt8));"
    )
    out, changed = patch.patch_source(FILTER_NEIGHBOR, old, new)
    assert changed and new in out and old not in out
    out2, changed2 = patch.patch_source(out, old, new)
    assert changed2 is False and out2 == out


def test_patch_source_missing_anchor_raises():
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.patch_source("nope", "old", "new")


def test_apply_to_tree_patches_all_reported_sites(tmp_path):
    root = tmp_path / "o-voxel"
    files = {
        "src/io/filter_neighbor.cpp": FILTER_NEIGHBOR,
        "src/io/filter_parent.cpp": FILTER_PARENT,
        "src/io/svo.cpp": SVO,
        "src/convert/flexible_dual_grid.cpp": (
            "torch::from_blob(voxels.data(), {int(voxels .size()), 3}, torch::kInt32).clone(),\n"
            "torch::from_blob(dual_vertices.data(), {int(dual_vertices.size()), 3}, torch::kFloat32).clone(),\n"
            "torch::from_blob(intersected.data(), {int(intersected.size()), 3}, torch::kBool).clone()\n"
        ),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    states = patch.apply_to_tree(root)
    assert all(s.startswith("PATCHED") for s in states)
    # Idempotent
    states2 = patch.apply_to_tree(root)
    assert all(s.startswith("APPLIED") for s in states2)

    neighbor = (root / "src/io/filter_neighbor.cpp").read_text(encoding="utf-8")
    assert "torch::zeros({N, C}" not in neighbor
    assert neighbor.count("static_cast<int64_t>(N)") == 2
    assert neighbor.count("static_cast<int64_t>(C)") == 2

    parent = (root / "src/io/filter_parent.cpp").read_text(encoding="utf-8")
    assert "static_cast<int64_t>(N_leaf)" in parent

    svo = (root / "src/io/svo.cpp").read_text(encoding="utf-8")
    assert "static_cast<int64_t>(svo.size())" in svo
    assert "static_cast<int64_t>(codes.size())" in svo

    dual = (root / "src/convert/flexible_dual_grid.cpp").read_text(encoding="utf-8")
    assert "int(voxels" not in dual
    assert "static_cast<int64_t>(voxels.size())" in dual


def test_replacements_cover_user_reported_files():
    rels = {rel for rel, _, _ in patch.REPLACEMENTS}
    assert "src/io/filter_neighbor.cpp" in rels
    assert "src/io/filter_parent.cpp" in rels
    assert "src/io/svo.cpp" in rels


def test_against_upstream_trellis_checkout_if_present():
    """When a TRELLIS.2 clone is available, every anchor must still match."""
    candidates = [
        Path("/tmp/i2l-ovoxel/o-voxel"),
        Path(__file__).resolve().parents[1] / "vendor" / "trellis2-cuda" / "o-voxel",
    ]
    root = next((p for p in candidates if p.is_dir()), None)
    if root is None:
        pytest.skip("no o-voxel checkout present")
    for rel, old, new in patch.REPLACEMENTS:
        path = root / rel
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert new in text or old in text, f"{path}: neither old nor new present"


def test_bootstrap_calls_the_narrowing_patch():
    import bootstrap_trellis_cuda as boot
    source = Path(boot.__file__).read_text(encoding="utf-8")
    assert "patch_ovoxel_msvc_narrowing" in source
