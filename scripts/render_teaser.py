#!/usr/bin/env -S uv run --script --python 3.13
# /// script
# requires-python = "==3.13.*"
# dependencies = ["robot-models", "bpy==5.2.2"]
# [tool.uv.sources]
# robot-models = { path = ".." }
# ///
"""Render labeled humanoid and hand lineups, following Body Models' Blender teaser.

uv run --script scripts/render_teaser.py --group all --device gpu
uv run --script scripts/render_teaser.py --group hands --models allegro shadow

Output: results/teasers/{humanoids,hands}.png. Meshes come from the NumPy API;
Blender is isolated in this Python 3.13 script environment, not a package dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import numpy as np
import trimesh
from mathutils import Vector

import robot_models
from robot_models import Joint

LINEUPS = {
    "humanoids": ("g1", "gr1", "h1", "t1", "talos"),
    "hands": ("ability", "allegro", "brainco", "inspire", "leap", "sharpa", "shadow"),
}
PASTELS = (
    (0.62, 0.78, 0.98, 1),
    (0.80, 0.70, 0.95, 1),
    (0.62, 0.88, 0.74, 1),
    (0.99, 0.73, 0.54, 1),
    (0.95, 0.63, 0.72, 1),
    (0.99, 0.88, 0.59, 1),
    (0.69, 0.86, 0.93, 1),
)
# Baseline Shadow variants retain source link names for display landmarks.
SHADOW_LANDMARKS = {
    "shadow-gendexgrasp": ("ffproximal", "ffmiddle", "lfproximal"),
    "shadow-dexgraspnet": ("robot0:ffproximal_child", "robot0:ffmiddle_child", "robot0:lfproximal_child"),
}
# Source grippers have no anatomical common-joint map. Axes are in the Y-up model frame.
GRIPPER_AXES = {
    "barrett-gendexgrasp": ((0, 0, 1), (0, 1, 0)),
    "robotiq-3finger-gendexgrasp": ((0, 0, 1), (0, 1, 0)),
    "ezgripper-gendexgrasp": ((1, 0, 0), (0, 0, 1)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", *LINEUPS), default="all")
    parser.add_argument("--models", nargs="+", choices=robot_models.list_models())
    parser.add_argument("--output-dir", type=Path, default=Path("results/teasers"))
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--background", choices=("transparent", "light", "dark"), default="transparent")
    parser.add_argument("--save-blend", action="store_true", help="Also save editable Blender scenes.")
    args = parser.parse_args()
    if args.models and args.group == "all":
        parser.error("--models requires --group humanoids or --group hands")
    if min(args.width, args.height, args.samples, args.threads) < 1:
        parser.error("width, height, samples and threads must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = LINEUPS if args.group == "all" else (args.group,)
    for group in groups:
        render_lineup(group, args.models or LINEUPS[group], args)


def render_lineup(group, names, args) -> None:
    print(f"Loading {group}: {', '.join(names)}", flush=True)
    meshes = [(name, display_mesh(name, group)) for name in names]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    height = 1.75 if group == "humanoids" else 1.0
    objects = add_lineup(meshes, height)
    configure_scene(args)
    camera = add_camera()
    labels = add_labels(names, objects, camera, height, args.background)
    fit_camera(camera, objects + labels, args.width / args.height)
    scene = bpy.context.scene
    scene.render.filepath = str(args.output_dir / f"{group}.png")
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir / f"{group}.blend"))
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {scene.render.filepath}", flush=True)


def display_mesh(name, group):
    model = robot_models.create_model(name, runtime="numpy")
    if model.has_hands != (group == "hands"):
        raise ValueError(f"{name} does not belong in the {group} lineup")
    params = model.get_rest_pose(hands="flat") if model.has_hands else model.get_tpose()
    if not model.has_hands:
        return model.forward_meshes(**params)[0]
    # Show the hand rather than the long forearm of forearm-rooted Shadow variants.
    links = model.forward_links(**params)
    parts = []
    for local, frame, joint in zip(model.link_meshes, links, model.link_joint_indices, strict=True):
        if model.joint_names[joint].lower().endswith("forearm"):
            continue
        part = local.copy()
        part.apply_transform(frame)
        parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    mesh.vertices = np.asarray(mesh.vertices) @ palm_basis(name, model, params)
    return mesh


def palm_basis(name, model, params):
    if name in GRIPPER_AXES:
        across, up = np.asarray(GRIPPER_AXES[name], dtype=float)
    else:
        if name in SHADOW_LANDMARKS:
            base, tip, outer = SHADOW_LANDMARKS[name]
        else:
            side = model.side
            joints = model.common_joints
            base = joints[Joint(f"{side}_index_mcp")]
            tip = joints.get(Joint(f"{side}_index_pip")) or joints[Joint(f"{side}_index_dip")]
            outer = joints.get(Joint(f"{side}_pinky_mcp")) or joints[Joint(f"{side}_ring_mcp")]
        frames = model.forward_skeleton(**params)
        positions = {joint: frame[:3, 3] for joint, frame in zip(model.joint_names, frames, strict=True)}
        across = positions[base] - positions[outer]
        up = positions[tip] - positions[base]
    up = up / np.linalg.norm(up)
    normal = np.cross(across, up)
    normal /= np.linalg.norm(normal)
    across = np.cross(up, normal)
    return np.stack((across, up, normal), axis=1)


def add_lineup(meshes, target_height):
    normalized = []
    for name, mesh in meshes:
        vertices = np.array(mesh.vertices, dtype=np.float32)
        vertices -= (vertices.min(0) + vertices.max(0)) / 2
        vertices *= target_height / np.ptp(vertices[:, 1])
        vertices[:, 1] -= vertices[:, 1].min()
        # Proper rotation from model Y-up to Blender Z-up; preserve face winding.
        vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        normalized.append((name, vertices, np.asarray(mesh.faces, dtype=np.int32)))
    gap = target_height * 0.26
    widths = [float(np.ptp(vertices[:, 0])) for _, vertices, _ in normalized]
    x = -(sum(widths) + gap * (len(widths) - 1)) / 2
    objects = []
    for index, ((name, vertices, faces), width) in enumerate(zip(normalized, widths, strict=True)):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.location.x = x + width / 2
        obj.data.materials.append(material(name, PASTELS[index % len(PASTELS)]))
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth_by_angle(angle=np.radians(35))
        obj.select_set(False)
        objects.append(obj)
        x += width + gap
    bpy.context.view_layer.update()
    return objects


def material(name, color, *, emission=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Metallic"].default_value = 0.08
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = 1
    return mat


def configure_scene(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.seed = 0
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    if args.device == "gpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        devices = [device for device in prefs.devices if device.type == "OPTIX"]
        if not devices:
            raise RuntimeError("No Cycles OptiX GPU found; use --device cpu")
        for device in prefs.devices:
            device.use = device in devices
        scene.cycles.device = "GPU"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = args.background == "transparent"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    color = (0.94, 0.95, 0.97, 1) if args.background == "light" else (0.035, 0.045, 0.065, 1)
    background.inputs["Color"].default_value = color
    background.inputs["Strength"].default_value = 0.35
    for name, rotation, energy, color in (
        ("Key", (45, 0, 25), 2.4, (1, 0.94, 0.88)),
        ("Fill", (50, 0, -130), 0.9, (0.72, 0.82, 1)),
    ):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        light.color = color
        light.angle = 0.15
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = tuple(np.radians(rotation))
        scene.collection.objects.link(obj)


def add_camera():
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.type = "ORTHO"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.location = (0, -12, 3.5)
    direction = Vector((0, 0, 0.85)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def add_labels(names, objects, camera, height, background):
    color = (0.07, 0.09, 0.13, 1) if background == "light" else (0.94, 0.95, 0.98, 1)
    ink = material("Labels", color, emission=True)
    labels = []
    bpy.context.view_layer.update()
    inverse = camera.matrix_world.inverted()
    for name, obj in zip(names, objects, strict=True):
        curve = bpy.data.curves.new(f"{name}_label", "FONT")
        curve.body = name.replace("-gendexgrasp", "\nGenDexGrasp").replace("-dexgraspnet", "\nDexGraspNet").upper()
        curve.align_x = "CENTER"
        curve.align_y = "BOTTOM_BASELINE"
        curve.size = height * 0.065
        label = bpy.data.objects.new(f"{name}_label", curve)
        bounds = np.asarray([inverse @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box])
        position = (bounds.min(0) + bounds.max(0)) / 2
        position[1] = bounds[:, 1].max() + height * 0.09
        label.location = camera.matrix_world @ Vector(position)
        label.rotation_euler = camera.rotation_euler
        label.data.materials.append(ink)
        bpy.context.scene.collection.objects.link(label)
        labels.append(label)
    bpy.context.view_layer.update()
    return labels


def fit_camera(camera, objects, aspect):
    bpy.context.view_layer.update()
    world_corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    inverse = camera.matrix_world.inverted()
    local = np.asarray([inverse @ point for point in world_corners])
    center = camera.matrix_world @ Vector((local.min(0) + local.max(0)) / 2)
    width, height = np.ptp(local[:, :2], axis=0)
    camera.data.ortho_scale = max(width, height * aspect) * 1.12
    direction = camera.rotation_euler.to_quaternion() @ Vector((0, 0, 1))
    camera.location = center + direction * 12


if __name__ == "__main__":
    main()
