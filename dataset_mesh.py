import json
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as data
import trimesh
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class MeshSample:
    mesh_path: str
    class_name: str
    class_id: int
    split: str
    sample_name: str
    render_dir: str
    depth_dir: str
    cache_npz: str


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    centroid = vertices.mean(axis=0, keepdims=True)
    vertices = vertices - centroid
    scale = np.linalg.norm(vertices, axis=1).max()
    if scale < 1e-6:
        return vertices
    return vertices / scale


def build_face_neighbors(triangles: np.ndarray) -> np.ndarray:
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_idx, tri in enumerate(triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for u, v in edges:
            key = tuple(sorted((int(u), int(v))))
            edge_to_faces.setdefault(key, []).append(face_idx)

    neighbors = np.zeros((triangles.shape[0], 3), dtype=np.int64)
    for face_idx, tri in enumerate(triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge_idx, (u, v) in enumerate(edges):
            key = tuple(sorted((int(u), int(v))))
            candidates = [idx for idx in edge_to_faces.get(key, []) if idx != face_idx]
            neighbors[face_idx, edge_idx] = candidates[0] if candidates else face_idx
    return neighbors


def deterministic_face_sample(num_faces: int, max_faces: int, key: str) -> np.ndarray:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_faces, size=max_faces, replace=False))


def process_mesh(mesh_path: str, cache_npz: str, max_faces: int = 500) -> Dict[str, np.ndarray]:
    if os.path.isfile(cache_npz):
        cached = np.load(cache_npz)
        return {key: cached[key] for key in cached.files}

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from {mesh_path}")
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    vertices = normalize_vertices(np.asarray(mesh.vertices, dtype=np.float32))
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

    original_num_faces = triangles.shape[0]
    if original_num_faces > max_faces:
        sampled_face_index = deterministic_face_sample(original_num_faces, max_faces, mesh_path)
        triangles = triangles[sampled_face_index]
        normals = normals[sampled_face_index]

    tri_vertices = vertices[triangles]
    centers = tri_vertices.mean(axis=1).astype(np.float32)
    corners = tri_vertices.reshape(tri_vertices.shape[0], 9).astype(np.float32)
    neighbors = build_face_neighbors(triangles)

    pad = max_faces - triangles.shape[0]
    valid_faces = np.zeros((max_faces,), dtype=np.float32)
    valid_faces[: triangles.shape[0]] = 1.0
    if pad > 0:
        centers = np.pad(centers, ((0, pad), (0, 0)))
        corners = np.pad(corners, ((0, pad), (0, 0)))
        normals = np.pad(normals, ((0, pad), (0, 0)))
        pad_neighbors = np.arange(triangles.shape[0], max_faces, dtype=np.int64)
        if triangles.shape[0] == 0:
            raise ValueError(f"Mesh {mesh_path} has zero faces")
        neighbors = np.pad(neighbors, ((0, pad), (0, 0)), constant_values=triangles.shape[0] - 1)
        for idx in pad_neighbors:
            neighbors[idx] = idx

    os.makedirs(os.path.dirname(cache_npz), exist_ok=True)
    np.savez_compressed(
        cache_npz,
        centers=centers.astype(np.float32),
        corners=corners.astype(np.float32),
        normals=normals.astype(np.float32),
        neighbors=neighbors.astype(np.int64),
        valid_faces=valid_faces.astype(np.float32),
        num_original_faces=np.array([original_num_faces], dtype=np.int32),
    )
    return process_mesh(mesh_path, cache_npz, max_faces=max_faces)


def _render_views_pyrender(
    mesh_path: str,
    render_dir: str,
    depth_dir: str,
    image_size: int,
    num_views: int,
    radius: float,
    platform: str,
):
    prev_platform = os.environ.get("PYOPENGL_PLATFORM")
    os.environ["PYOPENGL_PLATFORM"] = platform
    import pyrender

    tri_mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(tri_mesh, trimesh.Scene):
        tri_mesh = trimesh.util.concatenate(tuple(tri_mesh.geometry.values()))
    tri_mesh.vertices = normalize_vertices(np.asarray(tri_mesh.vertices, dtype=np.float32))
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)

    center = np.zeros(3, dtype=np.float32)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    for view_id in range(num_views):
        theta = 2.0 * np.pi * view_id / num_views
        eye = np.array([radius * np.cos(theta), radius * 0.35, radius * np.sin(theta)], dtype=np.float32)

        z_axis = eye - center
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye

        scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.2, 0.2, 0.2])
        scene.add(mesh)
        scene.add(camera, pose=pose)
        scene.add(light, pose=pose)

        color, depth = renderer.render(scene)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = (255.0 * depth / (depth.max() + 1e-6)).astype(np.uint8)
        depth_rgb = np.stack([depth, depth, depth], axis=-1)
        Image.fromarray(color).save(os.path.join(render_dir, f"view_{view_id:02d}.png"))
        Image.fromarray(depth_rgb).save(os.path.join(depth_dir, f"view_{view_id:02d}.png"))

    renderer.delete()
    if prev_platform is None:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ["PYOPENGL_PLATFORM"] = prev_platform


def _render_views_software(mesh_path: str, render_dir: str, depth_dir: str, image_size: int, num_views: int, radius: float):
    if os.path.isdir(render_dir) and os.path.isdir(depth_dir):
        render_files = [name for name in os.listdir(render_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]
        depth_files = [name for name in os.listdir(depth_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]
        if len(render_files) == num_views and len(depth_files) == num_views:
            return

    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    tri_mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(tri_mesh, trimesh.Scene):
        tri_mesh = trimesh.util.concatenate(tuple(tri_mesh.geometry.values()))
    vertices = normalize_vertices(np.asarray(tri_mesh.vertices, dtype=np.float32))
    faces = np.asarray(tri_mesh.faces, dtype=np.int64)
    tri_vertices = vertices[faces]
    face_normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

    center = np.zeros(3, dtype=np.float32)
    light_dir = np.array([0.3, 0.6, 1.0], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)

    for view_id in range(num_views):
        theta = 2.0 * np.pi * view_id / num_views
        eye = np.array([radius * np.cos(theta), radius * 0.35, radius * np.sin(theta)], dtype=np.float32)

        forward = center - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        true_up = np.cross(right, forward)
        view_rot = np.stack([right, true_up, forward], axis=0)

        camera_vertices = (vertices - eye) @ view_rot.T
        z = camera_vertices[:, 2]
        z = z - z.min() + 1.0
        x = camera_vertices[:, 0] / z
        y = camera_vertices[:, 1] / z
        projected = np.stack([x, y, z], axis=1)
        scale = image_size * 0.42
        projected[:, 0] = projected[:, 0] * scale + image_size / 2.0
        projected[:, 1] = -projected[:, 1] * scale + image_size / 2.0

        face_proj = projected[faces]
        face_depth = face_proj[:, :, 2].mean(axis=1)
        sort_index = np.argsort(face_depth)[::-1]

        color_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        depth_img = Image.new("L", (image_size, image_size), 0)
        color_draw = ImageDraw.Draw(color_img)
        depth_draw = ImageDraw.Draw(depth_img)

        depth_min = face_depth.min()
        depth_max = face_depth.max()

        for face_idx in sort_index:
            pts = [(float(face_proj[face_idx, i, 0]), float(face_proj[face_idx, i, 1])) for i in range(3)]
            shade = float(np.clip(np.dot(face_normals[face_idx], light_dir), 0.0, 1.0))
            shade = 0.25 + 0.75 * shade
            rgb = int(255 * shade)
            depth_value = int(255 * (face_depth[face_idx] - depth_min) / (depth_max - depth_min + 1e-8))
            depth_value = 255 - depth_value
            color_draw.polygon(pts, fill=(rgb, rgb, rgb))
            depth_draw.polygon(pts, fill=depth_value)

        color_img.save(os.path.join(render_dir, f"view_{view_id:02d}.png"))
        depth_rgb = Image.merge("RGB", (depth_img, depth_img, depth_img))
        depth_rgb.save(os.path.join(depth_dir, f"view_{view_id:02d}.png"))


def render_views(
    mesh_path: str,
    render_dir: str,
    depth_dir: str,
    image_size: int,
    num_views: int,
    radius: float = 2.0,
    backend: str = "pyrender_egl",
):
    if os.path.isdir(render_dir) and os.path.isdir(depth_dir):
        render_files = [name for name in os.listdir(render_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]
        depth_files = [name for name in os.listdir(depth_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]
        if len(render_files) == num_views and len(depth_files) == num_views:
            return

    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    if backend == "pyrender_egl":
        _render_views_pyrender(mesh_path, render_dir, depth_dir, image_size, num_views, radius, platform="egl")
    elif backend == "pyrender_x11":
        _render_views_pyrender(mesh_path, render_dir, depth_dir, image_size, num_views, radius, platform="xlib")
    elif backend == "software":
        _render_views_software(mesh_path, render_dir, depth_dir, image_size, num_views, radius)
    else:
        raise ValueError(f"Unsupported render backend: {backend}")


def discover_from_manifold40(data_root: str, cache_root: str, num_views: int) -> Tuple[List[str], Dict[str, List[MeshSample]]]:
    classnames = sorted(
        [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
    )
    class_to_idx = {name: idx for idx, name in enumerate(classnames)}
    samples = {"train": [], "test": []}
    for class_name in classnames:
        for split in ("train", "test"):
            split_dir = os.path.join(data_root, class_name, split)
            if not os.path.isdir(split_dir):
                continue
            for filename in sorted(os.listdir(split_dir)):
                if not filename.lower().endswith(".obj"):
                    continue
                mesh_path = os.path.join(split_dir, filename)
                sample_name = os.path.splitext(filename)[0]
                sample_cache_root = os.path.join(cache_root, class_name, split, sample_name)
                samples[split].append(
                    MeshSample(
                        mesh_path=mesh_path,
                        class_name=class_name,
                        class_id=class_to_idx[class_name],
                        split=split,
                        sample_name=sample_name,
                        render_dir=os.path.join(sample_cache_root, "render"),
                        depth_dir=os.path.join(sample_cache_root, "depth"),
                        cache_npz=os.path.join(sample_cache_root, "meshnet.npz"),
                    )
                )
    return classnames, samples


def discover_from_meta(meta_path: str, data_root: str, cache_root: str) -> Tuple[List[str], Dict[str, List[MeshSample]]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classnames = meta["classnames"]
    samples = {"train": [], "test": []}
    for split in ("train", "test"):
        for sample in meta[split]:
            mesh_path = sample["mesh_path"]
            if not os.path.isabs(mesh_path):
                mesh_path = os.path.join(data_root, mesh_path)
            render_dir = sample.get("render_dir")
            depth_dir = sample.get("depth_dir")
            sample_name = sample.get("sample_name", os.path.splitext(os.path.basename(mesh_path))[0])
            sample_cache_root = os.path.join(cache_root, sample["class_name"], split, sample_name)
            samples[split].append(
                MeshSample(
                    mesh_path=mesh_path,
                    class_name=sample["class_name"],
                    class_id=sample["class_id"],
                    split=split,
                    sample_name=sample_name,
                    render_dir=render_dir if render_dir is not None else os.path.join(sample_cache_root, "render"),
                    depth_dir=depth_dir if depth_dir is not None else os.path.join(sample_cache_root, "depth"),
                    cache_npz=sample.get("cache_npz", os.path.join(sample_cache_root, "meshnet.npz")),
                )
            )
    return classnames, samples


class MeshMultiViewDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform,
        image_size: int,
        cache_root: str,
        num_sampled_faces: int = 500,
        num_views: int = 9,
        meta_path: Optional[str] = None,
        render_on_the_fly: bool = True,
        render_backend: str = "pyrender_egl",
        load_views: bool = True,
    ):
        self.root = root
        self.transform = transform
        self.image_size = image_size
        self.cache_root = cache_root
        self.num_sampled_faces = num_sampled_faces
        self.num_views = num_views
        self.render_on_the_fly = render_on_the_fly
        self.render_backend = render_backend
        self.load_views = load_views

        if meta_path and os.path.isfile(meta_path):
            self.classnames, samples = discover_from_meta(meta_path, root, cache_root)
        else:
            self.classnames, samples = discover_from_manifold40(root, cache_root, num_views)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        self.samples = samples[split]

    def __len__(self):
        return len(self.samples)

    def _load_view_stack(self, directory: str) -> torch.Tensor:
        files = sorted([name for name in os.listdir(directory) if name.lower().endswith(IMAGE_EXTENSIONS)])
        if not files:
            raise ValueError(f"No view images found in {directory}")
        images = []
        for filename in files:
            img = Image.open(os.path.join(directory, filename)).convert("RGB")
            img = self.transform(img) if self.transform is not None else img
            images.append(img)
        return torch.stack(images)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        mesh_dict = process_mesh(sample.mesh_path, sample.cache_npz, max_faces=self.num_sampled_faces)
        mesh_inputs = {
            "centers": torch.from_numpy(mesh_dict["centers"]),
            "corners": torch.from_numpy(mesh_dict["corners"]),
            "normals": torch.from_numpy(mesh_dict["normals"]),
            "neighbors": torch.from_numpy(mesh_dict["neighbors"]),
            "valid_faces": torch.from_numpy(mesh_dict["valid_faces"]),
        }
        item = {
            "mesh_inputs": mesh_inputs,
            "class_id": torch.tensor(sample.class_id, dtype=torch.long),
            "class_name": sample.class_name,
            "mesh_path": sample.mesh_path,
        }
        if self.load_views:
            has_render_cache = (
                os.path.isdir(sample.render_dir)
                and os.path.isdir(sample.depth_dir)
                and len([name for name in os.listdir(sample.render_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]) == self.num_views
                and len([name for name in os.listdir(sample.depth_dir) if name.lower().endswith(IMAGE_EXTENSIONS)]) == self.num_views
            )
            if self.render_on_the_fly or not has_render_cache:
                render_views(
                    sample.mesh_path,
                    sample.render_dir,
                    sample.depth_dir,
                    image_size=self.image_size,
                    num_views=self.num_views,
                    backend=self.render_backend,
                )

            render_images = self._load_view_stack(sample.render_dir)
            depth_images = self._load_view_stack(sample.depth_dir)
            item["render_images"] = render_images
            item["depth_images"] = depth_images
            item["num_views"] = torch.tensor(render_images.shape[0], dtype=torch.long)
        return item
