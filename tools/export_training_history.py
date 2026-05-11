import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def mean_std(values):
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def load_history(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("training_history.json must be a JSON list.")
    return data


def export_csv(records, out_path: Path | None):
    field_order = [
        "timestamp",
        "dataset",
        "algorithm",
        "non_iid_alpha",
        "partition_data",
        "n_clients",
        "requested_n_clients",
        "iters",
        "wk_iters",
        "eval_every",
        "batch",
        "seed",
        "split_seed",
        "plan",
        "lr",
        "average_accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_duration_seconds",
        "memory_used",
        "mu",
        "model_momentum",
        "threshold",
        "lam",
    ]

    discovered_fields = set()
    for record in records:
        if isinstance(record, dict):
            discovered_fields.update(record.keys())

    fields = field_order + sorted(discovered_fields.difference(field_order))

    if out_path is None:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in records:
            if isinstance(record, dict):
                writer.writerow(record)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            if isinstance(record, dict):
                writer.writerow(record)


def print_markdown_aggregate(records, group_keys):
    groups = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        key = tuple(record.get(k) for k in group_keys)
        groups[key].append(record)

    header = group_keys + [
        "runs",
        "acc_mean±std",
        "f1_mean±std",
        "time_s_mean±std",
    ]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")

    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        acc_values = [safe_float(it.get("average_accuracy")) for it in items]
        f1_values = [safe_float(it.get("f1_score")) for it in items]
        time_values = [safe_float(it.get("training_duration_seconds")) for it in items]

        acc_m, acc_s = mean_std([v for v in acc_values if v is not None])
        f1_m, f1_s = mean_std([v for v in f1_values if v is not None])
        time_m, time_s = mean_std([v for v in time_values if v is not None])

        def fmt(m, s):
            if m is None:
                return "-"
            return f"{m:.4f}±{s:.4f}"

        row = list(key) + [
            str(len(items)),
            fmt(acc_m, acc_s),
            fmt(f1_m, f1_s),
            "-" if time_m is None else f"{time_m:.2f}±{time_s:.2f}",
        ]
        print("| " + " | ".join(str(x) for x in row) + " |")


def main():
    parser = argparse.ArgumentParser(
        description="Export and aggregate backend/training_history.json for paper tables."
    )
    parser.add_argument(
        "--history",
        type=str,
        default="backend/training_history.json",
        help="Path to training_history.json (default: backend/training_history.json).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Write a CSV export to this path (optional).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Print a Markdown aggregate table (mean±std) grouped by keys.",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="dataset,algorithm,non_iid_alpha,partition_data,n_clients,iters,wk_iters,eval_every",
        help="Comma-separated group keys for --aggregate.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    records = load_history(history_path)

    if args.out_csv:
        export_csv(records, Path(args.out_csv))

    if args.aggregate:
        group_keys = [k.strip() for k in args.group_by.split(",") if k.strip()]
        if not group_keys:
            raise ValueError("--group_by must include at least one key.")
        print_markdown_aggregate(records, group_keys)


if __name__ == "__main__":
    main()
