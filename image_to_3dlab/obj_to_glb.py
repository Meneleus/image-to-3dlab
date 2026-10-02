"""OBJ (+ MTL textures) → GLB without embedding ``bpy`` in a project venv.

Hunyuan paint's ``mesh_utils.convert_obj_to_glb`` used to ``import bpy`` at
module load. Finish already uses a real ``blender.exe`` (``I2L_BLENDER`` /
``find_blender``); this module shells out to that, then falls back to an
in-process ``bpy`` wheel if present, then ``trimesh``.
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

from image_to_3dlab.blender import find_blender

# Run inside Blender's own Python (has bpy). Mirrors upstream mesh_utils shading.
_BLENDER_SCRIPT = textwrap.dedent(
    """\
    import math
    import sys
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1 :]
    obj_path, glb_path, shade_type, angle_s, merge_s = argv
    angle = float(angle_s)
    merge = merge_s == "1"

    if "convert" not in bpy.data.scenes:
        bpy.data.scenes.new("convert")
    bpy.context.window.scene = bpy.data.scenes["convert"]
    for obj in list(bpy.context.scene.objects):
        obj.select_set(True)
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.obj_import(filepath=obj_path)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)

    if merge:
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles()
                bpy.ops.object.mode_set(mode="OBJECT")

    angle_rad = math.radians(angle)
    if shade_type == "FLAT":
        bpy.ops.object.shade_flat()
    elif shade_type == "AUTO_SMOOTH":
        if bpy.app.version < (4, 1, 0):
            bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=angle_rad)
        elif bpy.app.version < (4, 2, 0):
            bpy.ops.object.shade_smooth_by_angle(angle=angle_rad)
        else:
            bpy.ops.object.shade_auto_smooth(angle=angle_rad)
    else:
        bpy.ops.object.shade_smooth()

    bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)
    """
)


def convert_via_blender_exe(
    obj_path: str | Path,
    glb_path: str | Path,
    *,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
    blender: Path | None = None,
    run=subprocess.run,
) -> bool:
    """Headless ``blender.exe`` / ``I2L_BLENDER``. Returns True when glb exists."""
    exe = blender or find_blender()
    if exe is None:
        return False
    obj_path, glb_path = Path(obj_path), Path(glb_path)
    with tempfile.TemporaryDirectory(prefix="i2l-obj2glb-") as tmp:
        script = Path(tmp) / "convert.py"
        script.write_text(_BLENDER_SCRIPT, encoding="utf-8")
        result = run(
            [
                str(exe),
                "--background",
                "--python",
                str(script),
                "--",
                str(obj_path.resolve()),
                str(glb_path.resolve()),
                shade_type,
                str(auto_smooth_angle),
                "1" if merge_vertices else "0",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    return result.returncode == 0 and glb_path.is_file()


def convert_via_bpy(
    obj_path: str | Path,
    glb_path: str | Path,
    *,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """In-process pip ``bpy`` wheel when present (optional; not required)."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    # Re-use the same script body via Blender is awkward in-process; call ops directly.
    import math

    import bpy as _bpy

    obj_path, glb_path = str(obj_path), str(glb_path)
    if "convert" not in _bpy.data.scenes:
        _bpy.data.scenes.new("convert")
    _bpy.context.window.scene = _bpy.data.scenes["convert"]
    for obj in list(_bpy.context.scene.objects):
        obj.select_set(True)
        _bpy.data.objects.remove(obj, do_unlink=True)
    _bpy.ops.wm.obj_import(filepath=obj_path)
    _bpy.ops.object.select_all(action="DESELECT")
    for obj in _bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if merge_vertices:
        for obj in _bpy.context.selected_objects:
            if obj.type == "MESH":
                _bpy.context.view_layer.objects.active = obj
                _bpy.ops.object.mode_set(mode="EDIT")
                _bpy.ops.mesh.select_all(action="SELECT")
                _bpy.ops.mesh.remove_doubles()
                _bpy.ops.object.mode_set(mode="OBJECT")
    angle_rad = math.radians(auto_smooth_angle)
    if shade_type == "FLAT":
        _bpy.ops.object.shade_flat()
    elif shade_type == "AUTO_SMOOTH":
        if _bpy.app.version < (4, 1, 0):
            _bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=angle_rad)
        elif _bpy.app.version < (4, 2, 0):
            _bpy.ops.object.shade_smooth_by_angle(angle=angle_rad)
        else:
            _bpy.ops.object.shade_auto_smooth(angle=angle_rad)
    else:
        _bpy.ops.object.shade_smooth()
    _bpy.ops.export_scene.gltf(filepath=glb_path, use_active_scene=True)
    return Path(glb_path).is_file()


def convert_via_trimesh(obj_path: str | Path, glb_path: str | Path, **_: object) -> bool:
    """Last resort: trimesh (already a Hunyuan runtime dep)."""
    try:
        import trimesh
    except ImportError:
        return False
    loaded = trimesh.load(str(obj_path), force=None)
    loaded.export(str(glb_path))
    return Path(glb_path).is_file()


def convert_obj_to_glb(
    obj_path: str | Path,
    glb_path: str | Path,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
    """Prefer Finish's Blender, then pip bpy, then trimesh. Never requires User PATH."""
    kwargs = {
        "shade_type": shade_type,
        "auto_smooth_angle": auto_smooth_angle,
        "merge_vertices": merge_vertices,
    }
    for fn in (convert_via_blender_exe, convert_via_bpy, convert_via_trimesh):
        try:
            if fn(obj_path, glb_path, **kwargs):
                return True
        except Exception:  # noqa: BLE001, S112 — each route is best-effort
            continue
    return False
