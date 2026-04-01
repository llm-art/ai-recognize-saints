#!/usr/bin/env python3
"""
Two-part analysis of GPT generational regression on ICONCLASS:

Part 1 — Print-friendly confusion matrices (2 rows × 1 col):
    Row 1: GPT-4o Mini → GPT-5 Mini  (net improvement, but still some regressions)
    Row 2: GPT-4o (Nov) → GPT-5.2    (net regression)
    Each row is a confusion matrix: rows = true class, cols = wrong new-gen prediction.

Part 2 — Attribute overlap analysis:
    For the most confused pairs, extract ICONCLASS descriptions (test_2),
    parse attributes, compute Jaccard overlap, and visualise.

Outputs: dataset/shift_analysis/
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from itertools import combinations

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

TESTS = ["test_1", "test_2", "test_3"]

GPT_FAMILIES = [
    {
        "old": "gpt-4o-mini-2024-07-18",
        "new": "gpt-5-mini-2025-08-07",
        "label": "gpt-4o-mini-2024-07-18  →  gpt-5-mini-2025-08-07",
        "short": "mini",
    },
    {
        "old": "gpt-4o-2024-11-20",
        "new": "gpt-5.2-2025-12-11",
        "label": "gpt-4o-2024-11-20  →  gpt-5.2-2025-12-11",
        "short": "full",
    },
]


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_classes(dataset, test):
    path = BASE_DIR / "dataset" / f"{dataset}-data" / "classes.csv"
    df = pd.read_csv(path)
    col = "Label" if test in ("test_1", "test_3") else "Description"
    return list(df[["ID", col]].itertuples(index=False, name=None))


def load_ground_truth(dataset):
    path = BASE_DIR / "dataset" / f"{dataset}-data" / "2_ground_truth.json"
    with open(path) as f:
        data = json.load(f)
    return {item["item"]: item["class"] for item in data}


def load_preds(model, dataset, test, class_ids):
    model_dir = BASE_DIR / test / dataset / model
    probs_path = model_dir / "probs.npy"
    if not probs_path.exists():
        return {}
    probs = np.load(probs_path)
    ids_path = model_dir / "image_ids.txt"
    if ids_path.exists():
        images = ids_path.read_text().splitlines()
    else:
        images = (BASE_DIR / "dataset" / f"{dataset}-data" / "2_test.txt").read_text().splitlines()
    return {img: class_ids[int(probs[i].argmax())]
            for i, img in enumerate(images[:len(probs)])}


def short_saint(cid):
    """11H(PETER) → Peter, 11HH(MARY MAGDALENE) → Mary Magdalene."""
    m = re.search(r"\((.+)\)", cid)
    if m:
        name = m.group(1).title()
        # Shorten "Antony Abbot" etc.
        return name.replace("Mary Magdalene", "M. Magdalene")
    return cid


# ─── Part 1: Regression confusion chart ─────────────────────────────────────

def compute_regression_flows():
    """Aggregate regression flows across all 3 tests for both GPT families."""
    dataset = "ICONCLASS"
    gt = load_ground_truth(dataset)

    family_flows = {}
    for fam in GPT_FAMILIES:
        flow_counter = defaultdict(int)
        total_regressions = 0
        total_improvements = 0

        for test in TESTS:
            classes = load_classes(dataset, test)
            class_ids = [c[0] for c in classes]
            old_p = load_preds(fam["old"], dataset, test, class_ids)
            new_p = load_preds(fam["new"], dataset, test, class_ids)

            for img in set(old_p) & set(new_p):
                if img not in gt:
                    continue
                g = gt[img]
                old_ok = old_p[img] == g
                new_ok = new_p[img] == g
                if old_ok and not new_ok:
                    flow_counter[(g, new_p[img])] += 1
                    total_regressions += 1
                elif not old_ok and new_ok:
                    total_improvements += 1

        flow_df = (pd.DataFrame(
            [{"true_class": k[0], "wrong_pred": k[1], "count": v}
             for k, v in flow_counter.items()])
            .sort_values("count", ascending=False))

        family_flows[fam["short"]] = {
            "flows": flow_df,
            "label": fam["label"],
            "n_regressions": total_regressions,
            "n_improvements": total_improvements,
        }

    return family_flows


def plot_regression_chart(family_flows):
    """Two confusion matrices (one per GPT family) showing regression patterns.
    Rows = true class, Columns = what the new-gen model wrongly predicted.
    Only regression cases (old correct → new wrong) are counted."""
    out = OUTPUT_DIR / "gpt_regression_chart.pdf"

    # Get class order from data (consistent across both families)
    all_classes = set()
    for key in ("mini", "full"):
        flows = family_flows[key]["flows"]
        all_classes.update(flows["true_class"])
        all_classes.update(flows["wrong_pred"])
    class_order = sorted(all_classes)
    n_cls = len(class_order)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5),
                              gridspec_kw={"wspace": 0.45})

    cmaps = {"mini": "Oranges", "full": "Reds"}

    for ax, (key, fam_idx) in zip(axes, [("mini", 0), ("full", 1)]):
        data = family_flows[key]
        flows = data["flows"]
        n_reg = data["n_regressions"]
        n_imp = data["n_improvements"]
        net = n_imp - n_reg

        # Build confusion matrix: rows = true class, cols = wrong prediction
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for _, row in flows.iterrows():
            i = class_order.index(row["true_class"])
            j = class_order.index(row["wrong_pred"])
            cm[i, j] = row["count"]

        # Annotate: show count where > 0, empty string otherwise
        annot = np.where(cm > 0, cm.astype(str), "")

        sns.heatmap(cm, annot=annot, fmt="", cmap=cmaps[key],
                    xticklabels=class_order, yticklabels=class_order,
                    ax=ax, linewidths=0.6, linecolor="white",
                    cbar_kws={"shrink": 0.75},
                    vmin=0)

        new_model_label = data['label'].split('→')[-1].strip()
        ax.set_xlabel(f"Incorrect predictions of {new_model_label}", fontsize=10.5, fontfamily="serif")
        old_model_label = data['label'].split('→')[0].strip()
        ax.set_ylabel(f"Correct predictions of {old_model_label}", fontsize=10.5, fontfamily="serif")
        ax.tick_params(axis="both", labelsize=9)

        # Rotate x labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        net_str = f"+{net}" if net > 0 else str(net)
        ax.set_title(
            f"{data['label']}\n"
            f"{n_reg} losses, {n_imp} gains  (net {net_str})",
            fontsize=11, fontweight="bold", fontfamily="serif", pad=12)

    fig.suptitle("GPT Cross-Generation Misclassifications on ICONCLASS",
                 fontsize=13, fontweight="bold", fontfamily="serif", y=1.02)

    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Part 1] Chart → {out}")
    return family_flows


# ─── Part 2: Attribute overlap analysis ──────────────────────────────────────

def parse_attributes(description):
    """Extract attribute list from ICONCLASS description string."""
    # Pattern: "possible attributes: X, Y, Z"
    m = re.search(r"possible attributes?:\s*(.+)", description, re.IGNORECASE)
    if not m:
        return set()
    raw = m.group(1).rstrip(".")
    # Split on comma, clean up
    attrs = set()
    for a in raw.split(","):
        a = a.strip().lower()
        # Remove parenthetical qualifiers like "(upturned)" but keep content
        a = re.sub(r"[()]", "", a).strip()
        if a:
            attrs.add(a)
    return attrs


def analyze_attribute_overlap(family_flows):
    out_dir = OUTPUT_DIR / "attribute_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load descriptions (test_2 uses Description column)
    classes = load_classes("ICONCLASS", "test_2")
    class_desc = {c[0]: c[1] for c in classes}
    class_attrs = {cid: parse_attributes(desc) for cid, desc in class_desc.items()}

    print("\n[Part 2] Parsed attributes per saint:")
    for cid, attrs in class_attrs.items():
        print(f"  {short_saint(cid):20s}: {', '.join(sorted(attrs))}")

    # Collect all confused pairs across both families
    confused_pairs = defaultdict(int)
    for key in ("mini", "full"):
        for _, row in family_flows[key]["flows"].iterrows():
            pair = tuple(sorted([row["true_class"], row["wrong_pred"]]))
            confused_pairs[pair] += row["count"]

    # Build pairwise attribute overlap table
    all_class_ids = sorted(class_attrs.keys())
    records = []
    for c1, c2 in combinations(all_class_ids, 2):
        a1 = class_attrs[c1]
        a2 = class_attrs[c2]
        shared = a1 & a2
        union = a1 | a2
        jaccard = len(shared) / len(union) * 100 if union else 0
        pair_key = tuple(sorted([c1, c2]))
        confusion_count = confused_pairs.get(pair_key, 0)
        records.append({
            "saint_1": c1,
            "saint_2": c2,
            "label_1": short_saint(c1),
            "label_2": short_saint(c2),
            "attrs_1": ", ".join(sorted(a1)),
            "attrs_2": ", ".join(sorted(a2)),
            "shared_attrs": ", ".join(sorted(shared)) if shared else "(none)",
            "n_shared": len(shared),
            "n_union": len(union),
            "jaccard_%": round(jaccard, 1),
            "confusion_count": confusion_count,
        })

    pair_df = pd.DataFrame(records).sort_values("confusion_count", ascending=False)
    pair_df.to_csv(out_dir / "attribute_overlap_all_pairs.csv", index=False)

    # ── Jaccard heatmap (full 10×10) ──
    n = len(all_class_ids)
    jaccard_matrix = np.zeros((n, n))
    confusion_matrix = np.zeros((n, n))
    for r in records:
        i = all_class_ids.index(r["saint_1"])
        j = all_class_ids.index(r["saint_2"])
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = r["jaccard_%"]
        confusion_matrix[i, j] = confusion_matrix[j, i] = r["confusion_count"]
    np.fill_diagonal(jaccard_matrix, 100)

    short_labels = [short_saint(c) for c in all_class_ids]

    # ── Combined figure: Jaccard heatmap + scatter ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35})
    fig.suptitle("ICONCLASS Attribute Overlap vs GPT Confusion",
                 fontsize=14, fontweight="bold", fontfamily="serif")

    # Left: Jaccard heatmap
    ax = axes[0]
    mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool), k=1)
    sns.heatmap(jaccard_matrix, mask=mask, annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=short_labels, yticklabels=short_labels,
                ax=ax, linewidths=0.5, linecolor="#eee",
                cbar_kws={"label": "Jaccard similarity (%)", "shrink": 0.8},
                vmin=0, vmax=60)
    ax.set_title("Attribute Jaccard Similarity", fontsize=11.5,
                 fontweight="bold", fontfamily="serif")
    ax.tick_params(axis="both", labelsize=9)

    # Right: scatter — Jaccard vs confusion count
    ax = axes[1]
    scatter_df = pair_df[pair_df["confusion_count"] > 0].copy()

    ax.scatter(scatter_df["jaccard_%"], scatter_df["confusion_count"],
               s=scatter_df["confusion_count"] * 4 + 30,
               c="#C0392B", alpha=0.6, edgecolors="white", linewidth=0.5, zorder=3)

    # Annotate top confused pairs
    for _, row in scatter_df.head(10).iterrows():
        ax.annotate(f"{row['label_1']}–{row['label_2']}",
                    (row["jaccard_%"], row["confusion_count"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, fontfamily="serif", alpha=0.85)

    # Also annotate high-Jaccard pairs even if low confusion
    high_jaccard = pair_df[(pair_df["jaccard_%"] > 25) & (pair_df["confusion_count"] == 0)]
    for _, row in high_jaccard.iterrows():
        ax.annotate(f"{row['label_1']}–{row['label_2']}",
                    (row["jaccard_%"], row["confusion_count"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, fontfamily="serif", alpha=0.6, color="#666")

    ax.set_xlabel("Attribute Jaccard Similarity (%)", fontsize=10.5)
    ax.set_ylabel("GPT confusion count (both families, 3 tests)", fontsize=10.5)
    ax.set_title("Does Attribute Overlap Predict Confusion?", fontsize=11.5,
                 fontweight="bold", fontfamily="serif")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Correlation annotation
    if len(scatter_df) > 2:
        from scipy.stats import spearmanr
        all_pairs_for_corr = pair_df.copy()
        rho, p = spearmanr(all_pairs_for_corr["jaccard_%"],
                           all_pairs_for_corr["confusion_count"])
        ax.text(0.97, 0.97,
                f"Spearman ρ = {rho:.2f}\np = {p:.3f}" +
                (" *" if p < 0.05 else " (n.s.)"),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, fontfamily="serif",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))

    plt.savefig(out_dir / "attribute_overlap_vs_confusion.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "attribute_overlap_vs_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Detailed table: top confused pairs with shared attributes ──
    top_confused = pair_df[pair_df["confusion_count"] > 0].head(15)
    print("\n[Part 2] Top confused pairs and their shared attributes:")
    print(f"  {'Pair':<40s} {'Confusions':>10s} {'Jaccard':>8s}   Shared attributes")
    print("  " + "─" * 100)
    for _, row in top_confused.iterrows():
        pair_str = f"{short_saint(row['saint_1'])} ↔ {short_saint(row['saint_2'])}"
        print(f"  {pair_str:<40s} {row['confusion_count']:>10d} {row['jaccard_%']:>7.1f}%   {row['shared_attrs']}")

    # ── Per-saint attribute comparison card for top 6 confused pairs ──
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Attribute Comparison for Most Confused Saint Pairs\n"
                 "(ICONCLASS descriptions — shared attributes highlighted)",
                 fontsize=13, fontweight="bold", fontfamily="serif")

    top6 = top_confused.head(6)
    for idx, (ax, (_, row)) in enumerate(zip(axes.flat, top6.iterrows())):
        a1 = class_attrs[row["saint_1"]]
        a2 = class_attrs[row["saint_2"]]
        shared = a1 & a2
        only_1 = a1 - a2
        only_2 = a2 - a1

        # Build text blocks
        s1_name = short_saint(row["saint_1"])
        s2_name = short_saint(row["saint_2"])

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_axis_off()

        # Title
        ax.text(5, 9.5, f"{s1_name}  ↔  {s2_name}",
                ha="center", fontsize=11, fontweight="bold", fontfamily="serif")
        ax.text(5, 8.8,
                f"{row['confusion_count']} confusions  |  Jaccard = {row['jaccard_%']:.0f}%",
                ha="center", fontsize=9, fontfamily="serif", color="#666")

        # Three columns: Saint 1 only | Shared | Saint 2 only
        col_x = [1.2, 5, 8.8]
        headers = [f"Only {s1_name}", "Shared", f"Only {s2_name}"]
        attr_lists = [sorted(only_1), sorted(shared), sorted(only_2)]
        header_colors = ["#3498DB", "#E74C3C", "#3498DB"]

        for cx, header, attrs, hcolor in zip(col_x, headers, attr_lists, header_colors):
            ax.text(cx, 7.8, header, ha="center", fontsize=9, fontweight="bold",
                    fontfamily="serif", color=hcolor)
            for j, attr in enumerate(attrs):
                y = 7.0 - j * 0.65
                if y < 0.5:
                    ax.text(cx, y + 0.3, f"... +{len(attrs) - j} more",
                            ha="center", fontsize=8, fontfamily="serif", color="#999")
                    break
                fc = "#FADBD8" if header == "Shared" else "#EBF5FB"
                ax.text(cx, y, attr, ha="center", fontsize=8.5, fontfamily="serif",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=fc,
                                  edgecolor="#ddd", alpha=0.8))

        # Separator lines
        ax.axhline(y=8.4, xmin=0.05, xmax=0.95, color="#ddd", linewidth=0.8)

    # Remove unused axes
    for ax in axes.flat[len(top6):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "attribute_comparison_cards.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "attribute_comparison_cards.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[Part 2] Attribute analysis → {out_dir}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Part 1: Regression confusion chart")
    print("=" * 60)
    family_flows = compute_regression_flows()
    plot_regression_chart(family_flows)

    print("\n" + "=" * 60)
    print("  Part 2: Attribute overlap analysis")
    print("=" * 60)
    analyze_attribute_overlap(family_flows)

    print(f"\nDone. All outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
