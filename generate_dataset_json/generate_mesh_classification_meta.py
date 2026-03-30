import argparse
import json
import os


def build_split(dataset_root, split, class_to_idx):
    samples = []
    for class_name in sorted(class_to_idx):
        split_dir = os.path.join(dataset_root, class_name, split)
        if not os.path.isdir(split_dir):
            continue
        for filename in sorted(os.listdir(split_dir)):
            if not filename.lower().endswith(".obj"):
                continue
            sample_name = os.path.splitext(filename)[0]
            samples.append(
                {
                    "mesh_path": os.path.join(class_name, split, filename),
                    "class_name": class_name,
                    "class_id": class_to_idx[class_name],
                    "sample_name": sample_name,
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser("Generate Manifold40 mesh classification metadata")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--seen_classnames", type=str, nargs="*", default=None)
    parser.add_argument("--unseen_classnames", type=str, nargs="*", default=None)
    args = parser.parse_args()

    classnames = sorted(
        [name for name in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, name))]
    )
    class_to_idx = {name: idx for idx, name in enumerate(classnames)}
    meta = {
        "classnames": classnames,
        "seen_classnames": args.seen_classnames,
        "unseen_classnames": args.unseen_classnames,
        "train": build_split(args.dataset_root, "train", class_to_idx),
        "test": build_split(args.dataset_root, "test", class_to_idx),
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
