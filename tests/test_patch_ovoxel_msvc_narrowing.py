"""MSVC o-voxel patch: real anchors, idempotent, no invented ``""d`` literals."""

from __future__ import annotations

from pathlib import Path

import patch_ovoxel_msvc_narrowing as patch
import pytest

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

# Upstream flexible_dual_grid.cpp sites that break MSVC (C3688 / C4838 / C2398).
FLEXIBLE_DUAL_GRID = '''
    if (segment_length < 1e-6d) continue; // Skip degenerate edges (zero-length)
    if (dir[axis] == 0.0d) {
        tMax[axis] = std::numeric_limits<double>::infinity();
    }
    int4 quad_indices{i, neigh_indices[0], neigh_indices[2], neigh_indices[1]};
    int4 quad_indices{i, neigh_indices[1], neigh_indices[5], neigh_indices[3]};
    int4 quad_indices{i, neigh_indices[0], neigh_indices[4], neigh_indices[3]};
    torch::from_blob(voxels.data(), {int(voxels .size()), 3}, torch::kInt32).clone(),
    torch::from_blob(dual_vertices.data(), {int(dual_vertices.size()), 3}, torch::kFloat32).clone(),
    torch::from_blob(intersected.data(), {int(intersected.size()), 3}, torch::kBool).clone()
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


def test_patch_source_strips_d_float_suffix_idempotently():
    old = "if (segment_length < 1e-6d) continue; // Skip degenerate edges (zero-length)"
    new = "if (segment_length < 1e-6) continue; // Skip degenerate edges (zero-length)"
    out, changed = patch.patch_source(FLEXIBLE_DUAL_GRID, old, new)
    assert changed
    assert "1e-6d" not in out
    assert "1e-6" in out
    out2, changed2 = patch.patch_source(out, old, new)
    assert changed2 is False and out2 == out
    # Must not invent a user-defined literal.
    assert '""d' not in out
    assert "operator" not in out


def test_patch_source_missing_anchor_raises():
    with pytest.raises(RuntimeError, match="anchor missing"):
        patch.patch_source("nope", "old", "new")


def test_apply_to_tree_patches_all_reported_sites(tmp_path):
    root = tmp_path / "o-voxel"
    files = {
        "src/io/filter_neighbor.cpp": FILTER_NEIGHBOR,
        "src/io/filter_parent.cpp": FILTER_PARENT,
        "src/io/svo.cpp": SVO,
        "src/convert/flexible_dual_grid.cpp": FLEXIBLE_DUAL_GRID,
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
    assert "1e-6d" not in dual
    assert "0.0d" not in dual
    assert '""d' not in dual
    assert "if (segment_length < 1e-6) continue" in dual
    assert "if (dir[axis] == 0.0) {" in dual
    assert "static_cast<int>(neigh_indices[0])" in dual
    assert "static_cast<int>(neigh_indices[5])" in dual
    assert "int(voxels" not in dual
    assert "static_cast<int64_t>(voxels.size())" in dual
    # Unpatched brace-init must be gone.
    assert "int4 quad_indices{i, neigh_indices[" not in dual


def test_replacements_cover_user_reported_files():
    rels = {rel for rel, _, _ in patch.REPLACEMENTS}
    assert "src/io/filter_neighbor.cpp" in rels
    assert "src/io/filter_parent.cpp" in rels
    assert "src/io/svo.cpp" in rels
    assert "src/convert/flexible_dual_grid.cpp" in rels


def test_replacements_new_snippets_never_add_d_float_suffix():
    for rel, old, new in patch.REPLACEMENTS:
        assert patch._BAD_DOUBLE_SUFFIX.search(new) is None, (rel, new)
        # Casting sites stay exact anchors — no token-splitting near literals.
        if "1e-6" in old or "0.0d" in old:
            assert "static_cast" not in new


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
