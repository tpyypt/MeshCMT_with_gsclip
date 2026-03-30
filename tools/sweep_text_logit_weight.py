import argparse
import json
import os
import re
import subprocess
import sys


BRANCH_PATTERN = re.compile(r"^(geometry_only|direct_only|visual_only|combined)\s*$")
TABLE_ROW_PATTERN = re.compile(r"^\|\s*([a-zA-Z0-9_]+)\s*\|\s*([0-9.]+)\s*\|$")


def parse_eval_log(log_path):
    branch = None
    metrics = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            branch_match = BRANCH_PATTERN.match(line)
            if branch_match:
                branch = branch_match.group(1)
                metrics.setdefault(branch, {})
                continue
            row_match = TABLE_ROW_PATTERN.match(line)
            if row_match and branch is not None:
                metric_name = row_match.group(1)
                metric_value = float(row_match.group(2))
                metrics[branch][metric_name] = metric_value
    if "visual_only" not in metrics:
        raise ValueError(f"failed to parse eval metrics from {log_path}")
    return metrics


def run_single_eval(args, weight):
    run_name = f"w{str(weight).replace('.', '')}"
    save_path = os.path.join(args.save_root, run_name)
    cmd = [
        sys.executable,
        "test_mesh_cls.py",
        "--data_root",
        args.data_root,
        "--save_path",
        save_path,
        "--stage2_checkpoint_path",
        args.stage2_checkpoint_path,
        "--cache_root",
        args.cache_root,
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--text_logit_weight",
        str(weight),
        "--mesh_mask_ratio",
        str(args.mesh_mask_ratio),
        "--render_backend",
        args.render_backend,
        "--split",
        args.split,
        "--disable_render_cache_generation",
    ]
    cmd.extend(["--features_list", *[str(value) for value in args.features_list]])
    if args.meta_path:
        cmd.extend(["--meta_path", args.meta_path])
    if args.use_depth_branch:
        cmd.append("--use_depth_branch")
    else:
        cmd.append("--no-use_depth_branch")

    subprocess.run(cmd, cwd=args.project_root, check=True)
    log_path = os.path.join(save_path, "log.txt")
    metrics = parse_eval_log(log_path)
    return {
        "weight": weight,
        "save_path": save_path,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser("Sweep text_logit_weight and summarize eval metrics")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--stage2_checkpoint_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="./.cache/manifold40")
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--features_list", type=int, nargs="+", default=[24])
    parser.add_argument("--mesh_mask_ratio", type=float, default=0.0)
    parser.add_argument("--render_backend", type=str, default="pyrender_egl")
    parser.add_argument("--use_depth_branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    results = [run_single_eval(args, weight) for weight in args.weights]
    summary_path = os.path.join(args.save_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_visual = max(results, key=lambda item: item["metrics"]["visual_only"]["accuracy"])
    best_combined = max(results, key=lambda item: item["metrics"]["combined"]["accuracy"])

    print(f"summary_saved={summary_path}")
    print(
        "best_visual_only="
        f"weight:{best_visual['weight']}, accuracy:{best_visual['metrics']['visual_only']['accuracy']:.2f}, "
        f"macro_f1:{best_visual['metrics']['visual_only'].get('macro_f1', 0.0):.2f}, save_path:{best_visual['save_path']}"
    )
    print(
        "best_combined="
        f"weight:{best_combined['weight']}, accuracy:{best_combined['metrics']['combined']['accuracy']:.2f}, "
        f"macro_f1:{best_combined['metrics']['combined'].get('macro_f1', 0.0):.2f}, save_path:{best_combined['save_path']}"
    )
    for result in results:
        visual_acc = result["metrics"]["visual_only"]["accuracy"]
        combined_acc = result["metrics"]["combined"]["accuracy"]
        direct_acc = result["metrics"]["direct_only"]["accuracy"]
        print(
            f"weight={result['weight']}: visual_only={visual_acc:.2f}, "
            f"combined={combined_acc:.2f}, direct_only={direct_acc:.2f}"
        )


if __name__ == "__main__":
    main()
