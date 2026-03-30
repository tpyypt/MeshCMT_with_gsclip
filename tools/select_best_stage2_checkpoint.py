import argparse
import os
import re


PATTERN = re.compile(
    r"epoch \[(?P<epoch>\d+)/(?P<total>\d+)\], .*?"
    r"direct_acc: (?P<direct_acc>[0-9.]+), "
    r"visual_text_acc: (?P<visual_text_acc>[0-9.]+), "
    r"combined_acc: (?P<combined_acc>[0-9.]+), "
    r"geometry_teacher_acc: (?P<geometry_teacher_acc>[0-9.]+)"
)


def parse_log(log_path):
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = PATTERN.search(line)
            if not match:
                continue
            record = {"epoch": int(match.group("epoch"))}
            for key in ("direct_acc", "visual_text_acc", "combined_acc", "geometry_teacher_acc"):
                record[key] = float(match.group(key))
            rows.append(record)
    if not rows:
        raise ValueError(f"no stage2 epoch summary found in {log_path}")
    return rows


def main():
    parser = argparse.ArgumentParser("Select the best stage2 checkpoint from log.txt")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="visual_text_acc",
        choices=["direct_acc", "visual_text_acc", "combined_acc", "geometry_teacher_acc"],
    )
    parser.add_argument("--stage2_dir", type=str, default=None)
    args = parser.parse_args()

    rows = parse_log(args.log_path)
    best = max(rows, key=lambda row: row[args.metric])
    stage2_dir = args.stage2_dir or os.path.dirname(args.log_path)
    checkpoint_path = os.path.join(stage2_dir, f"epoch_{best['epoch']}.pth")

    print(f"best_metric={args.metric}")
    print(f"best_epoch={best['epoch']}")
    print(f"best_value={best[args.metric]:.4f}")
    print(f"checkpoint_path={checkpoint_path}")
    print("all_metrics=" + ", ".join(f"{key}={best[key]:.4f}" for key in ("direct_acc", "visual_text_acc", "combined_acc", "geometry_teacher_acc")))


if __name__ == "__main__":
    main()
