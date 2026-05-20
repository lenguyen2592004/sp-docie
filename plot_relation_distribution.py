#!/usr/bin/env python3
"""
Visualize relation-label distributions across DocRE datasets.

Default inputs:
- train_distant_clean.json
- train_annotated.json
- dev.json
- inference_results/moe_train_20260317_002320_4a158be2_result.json
- rel_info.json

Outputs:
- relation_counts_wide.csv
- relation_counts_long.csv
- relation_distribution_topk_count.png
- relation_distribution_topk_ratio.png
"""

import argparse
import csv
import json
import os
from collections import Counter


def load_json_robust(path):
    """Load JSON with fallback for partially truncated arrays."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: clip to last closing bracket and try again.
        end = text.rfind("]")
        if end == -1:
            raise
        clipped = text[: end + 1]
        return json.loads(clipped)


def relation_counts(data):
    counter = Counter()
    for item in data:
        if not isinstance(item, dict):
            continue
        # DocRED train/dev format: each item has labels list.
        if "labels" in item and isinstance(item.get("labels"), list):
            for lbl in item.get("labels", []):
                if not isinstance(lbl, dict):
                    continue
                rel = lbl.get("r")
                if rel is not None:
                    counter[str(rel)] += 1
            continue

        # Inference output format: each item is already one predicted fact.
        rel = item.get("r")
        if rel is not None:
            counter[str(rel)] += 1
    return counter


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csvs(out_dir, rel_names, dataset_names, counters):
    rel_ids = sorted(rel_names.keys())

    wide_path = os.path.join(out_dir, "relation_counts_wide.csv")
    with open(wide_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relation_id", "relation_name"] + dataset_names)
        for rid in rel_ids:
            row = [rid, rel_names[rid]] + [counters[name].get(rid, 0) for name in dataset_names]
            w.writerow(row)

    long_path = os.path.join(out_dir, "relation_counts_long.csv")
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "relation_id", "relation_name", "count", "ratio"])
        totals = {name: sum(counters[name].values()) for name in dataset_names}
        for name in dataset_names:
            total = max(1, totals[name])
            for rid in rel_ids:
                c = counters[name].get(rid, 0)
                w.writerow([name, rid, rel_names[rid], c, c / total])

    return wide_path, long_path


def build_topk_relation_ids(rel_names, dataset_names, counters, top_k):
    scores = []
    for rid in rel_names.keys():
        s = sum(counters[name].get(rid, 0) for name in dataset_names)
        scores.append((s, rid))
    scores.sort(reverse=True)
    top = [rid for _, rid in scores[:top_k]]
    return top


def plot_grouped_bar(out_path, title, xlabels, series_dict, ylabel):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    n_groups = len(xlabels)
    n_series = len(series_dict)
    if n_groups == 0 or n_series == 0:
        raise ValueError("No data to plot")

    width = 0.8 / n_series
    x = list(range(n_groups))

    plt.figure(figsize=(max(14, n_groups * 0.6), 8))
    for i, (name, vals) in enumerate(series_dict.items()):
        xi = [v + (i - (n_series - 1) / 2.0) * width for v in x]
        plt.bar(xi, vals, width=width, label=name)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, xlabels, rotation=70, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot relation label distributions for DocRE datasets")
    parser.add_argument("--train-distant-clean", default="train_distant_clean.json")
    parser.add_argument("--train-annotated", default="train_annotated.json")
    parser.add_argument("--dev", default="dev.json")
    parser.add_argument(
        "--predict",
        default="inference_results/moe_train_20260317_002320_4a158be2_result.json",
        help="Path to flat prediction JSON where each item has field r",
    )
    parser.add_argument("--rel-info", default="rel_info.json")
    parser.add_argument("--out-dir", default="analysis/relation_distribution")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K relations by total count across all sets")
    args = parser.parse_args()

    rel_names = load_json_robust(args.rel_info)
    if not isinstance(rel_names, dict):
        raise ValueError("rel_info.json must be a JSON object mapping relation_id -> relation_name")

    dataset_files = {
        "train_distant_clean": args.train_distant_clean,
        "train_annotated": args.train_annotated,
        "dev": args.dev,
        "predict": args.predict,
    }

    ensure_output_dir(args.out_dir)

    counters = {}
    totals = {}
    for ds_name, path in dataset_files.items():
        data = load_json_robust(path)
        counters[ds_name] = relation_counts(data)
        totals[ds_name] = sum(counters[ds_name].values())

    dataset_names = list(dataset_files.keys())

    wide_csv, long_csv = write_csvs(args.out_dir, rel_names, dataset_names, counters)

    top_rel_ids = build_topk_relation_ids(
        rel_names=rel_names,
        dataset_names=dataset_names,
        counters=counters,
        top_k=max(1, int(args.top_k)),
    )

    xlabels = [f"{rid} | {rel_names.get(rid, rid)}" for rid in top_rel_ids]

    count_series = {
        ds: [counters[ds].get(rid, 0) for rid in top_rel_ids]
        for ds in dataset_names
    }
    count_png = os.path.join(args.out_dir, "relation_distribution_topk_count.png")
    ratio_png = os.path.join(args.out_dir, "relation_distribution_topk_ratio.png")

    plot_ok = True
    plot_err = None
    try:
        plot_grouped_bar(
            out_path=count_png,
            title=f"Top-{len(top_rel_ids)} Relation Distribution (Absolute Count)",
            xlabels=xlabels,
            series_dict=count_series,
            ylabel="Count",
        )
    except Exception as e:
        plot_ok = False
        plot_err = e

    ratio_series = {}
    for ds in dataset_names:
        total = max(1, totals[ds])
        ratio_series[ds] = [counters[ds].get(rid, 0) / total for rid in top_rel_ids]

    if plot_ok:
        try:
            plot_grouped_bar(
                out_path=ratio_png,
                title=f"Top-{len(top_rel_ids)} Relation Distribution (Ratio in Dataset)",
                xlabels=xlabels,
                series_dict=ratio_series,
                ylabel="Ratio",
            )
        except Exception as e:
            plot_ok = False
            plot_err = e

    print("Done.")
    print(f"- Wide CSV : {wide_csv}")
    print(f"- Long CSV : {long_csv}")
    if plot_ok:
        print(f"- Count PNG: {count_png}")
        print(f"- Ratio PNG: {ratio_png}")
    else:
        print("- Plot PNG : skipped")
        print(f"  Reason   : {plot_err}")
        print("  Tip      : pip install matplotlib")
    for ds in dataset_names:
        print(f"- {ds} total labels: {totals[ds]}")


if __name__ == "__main__":
    main()
