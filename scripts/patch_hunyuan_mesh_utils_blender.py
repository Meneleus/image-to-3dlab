#!/usr/bin/env python3
"""Stop Hunyuan paint requiring pip ``bpy`` in the vendor venv.

Upstream ``hy3dpaint/DifferentiableRenderer/mesh_utils.py`` does ``import bpy`` at
module load so ``textureGenPipeline`` cannot import on Windows (Finish uses
``blender.exe`` / ``I2L_BLENDER``, not a bpy wheel). This removes the hard import
and routes ``convert_obj_to_glb`` through ``image_to_3dlab.obj_to_glb``
(blender.exe → bpy → trimesh). Idempotent.

    python scripts/patch_hunyuan_mesh_utils_blender.py
    python scripts/patch_hunyuan_mesh_utils_blender.py \\
        --root vendor/hunyuan3d-cuda/hy3dpaint/DifferentiableRenderer
    python scripts/patch_hunyuan_mesh_utils_blender.py --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "hunyuan3d-cuda" / "hy3dpaint" / "DifferentiableRenderer"
TARGET = "mesh_utils.py"
MARKER = "image-to-3dlab: obj_to_glb without pip bpy"

IMPORT_OLD = "import bpy\n"
IMPORT_NEW = f"# {MARKER}\n"

CONVERT_OLD = '''\
def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """Convert OBJ file to GLB format using Blender."""
    try:
        _setup_blender_scene()
        _clear_scene_objects()

        # Import OBJ file
        bpy.ops.wm.obj_import(filepath=obj_path)
        _select_mesh_objects()

        # Process meshes
        _merge_vertices_if_needed(merge_vertices)
        _apply_shading(shade_type, auto_smooth_angle)

        # Export to GLB
        bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)
        return True
    except Exception:
        return False
'''

CONVERT_NEW = '''\
def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """Convert OBJ file to GLB format using Blender."""
    # image-to-3dlab: Finish's blender.exe / I2L_BLENDER — no pip bpy in this venv
    try:
        from image_to_3dlab.obj_to_glb import convert_obj_to_glb as _i2l_convert
        return _i2l_convert(
            obj_path,
            glb_path,
            shade_type=shade_type,
            auto_smooth_angle=auto_smooth_angle,
            merge_vertices=merge_vertices,
        )
    except Exception:
        return False
'''


def apply(source: str) -> str:
    if MARKER in source and CONVERT_OLD not in source:
        return source
    if IMPORT_OLD not in source and "import bpy" in source.split("def convert_obj_to_glb", 1)[0]:
        # Unusual layout — still need a clear failure.
        raise SystemExit("anchor not found: expected top-level `import bpy` in mesh_utils.py")
    if CONVERT_OLD not in source and MARKER not in source:
        raise SystemExit("anchor not found: convert_obj_to_glb body in mesh_utils.py")
    text = source
    if IMPORT_OLD in text:
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    elif MARKER not in text:
        raise SystemExit("anchor not found: `import bpy` line")
    if CONVERT_OLD in text:
        text = text.replace(CONVERT_OLD, CONVERT_NEW, 1)
    return text


def is_patched(source: str) -> bool:
    return MARKER in source and CONVERT_OLD not in source and "import bpy\n" not in source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = args.root / TARGET if args.root.is_dir() else args.root
    if not path.is_file():
        print(f"MISSING {path}")
        return 2
    text = path.read_text(encoding="utf-8")
    if args.check:
        print(f"{'APPLIED' if is_patched(text) else 'ABSENT'} {path}")
        return 0 if is_patched(text) else 2
    new = apply(text)
    if new == text:
        print(f"APPLIED {path}")
        return 0
    path.write_text(new, encoding="utf-8")
    print(f"PATCHED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
