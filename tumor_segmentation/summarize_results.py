"""Collect the test results of all runs into one table.

    python summarize_results.py                      # scans every metrics.json below runs/<run>/
    python summarize_results.py --runs_dir /other/runs --csv results.csv
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["model", "dataset", "split", "run", "checkpoint", "num_cases", "num_slices",
           "slice_dice_mean", "slice_iou_mean", "case_dice_mean", "case_iou_mean", "dice3d_mean"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default=os.path.join(HERE, "runs"))
    parser.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "**", "metrics.json"), recursive=True)):
        with open(path) as f:
            m = json.load(f)
        rows.append(dict(m, run=os.path.relpath(os.path.dirname(path), args.runs_dir),
                         checkpoint=os.path.basename(m.get("checkpoint", ""))))
    if not rows:
        print(f"no metrics.json found under {args.runs_dir}")
        return
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["run"]))

    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c, "")) for c in COLUMNS) + " |")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
