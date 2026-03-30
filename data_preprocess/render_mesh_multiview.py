import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_mesh import discover_from_manifold40, render_views


def main():
    parser = argparse.ArgumentParser("Render Manifold40 meshes into cached multiview RGB/depth images")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--num_views", type=int, default=9)
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--render_backend", type=str, default="pyrender_egl", choices=["pyrender_egl", "pyrender_x11", "software"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _, samples = discover_from_manifold40(args.data_root, args.cache_root, args.num_views)
    splits = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits:
        for sample in samples[split]:
            render_views(
                sample.mesh_path,
                sample.render_dir,
                sample.depth_dir,
                image_size=args.image_size,
                num_views=args.num_views,
                backend=args.render_backend,
            )
            if args.verbose:
                print(f"rendered {sample.mesh_path}")


if __name__ == "__main__":
    main()
