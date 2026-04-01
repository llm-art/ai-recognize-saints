#!/usr/bin/env python3
"""
Analyze consistency trends across model generations.

Investigates why next-generation LLMs achieve higher cross-dataset consistency
than legacy LLMs and contrastive baselines. Produces:

1. Generation-level consistency comparison (box + bar plots)
2. Variance analysis: newer models have lower cross-test variance
3. Per-class consistency breakdown: which classes drive the gap?
4. Agreement-with-ground-truth analysis: consistency through correctness
5. Pairwise successor improvement waterfall
6. Statistical significance tests (Mann-Whitney U)

Outputs saved to: dataset/consistency_trends/
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

BASE_DIR = Path(__file__).parent.parent
CONSISTENCY_DIR = BASE_DIR / "dataset" / "consistency"
OUTPUT_DIR = BASE_DIR / "dataset" / "consistency_trends"
TESTS = ["test_1", "test_2", "test_3"]

# ─── Model groups ────────────────────────────────────────────────────────────
CONTRASTIVE = [
    "clip-vit-base-patch32",
    "clip-vit-base-patch16",
    "clip-vit-large-patch14",
    "siglip-base-patch16-512",
    "siglip-large-patch16-384",
    "siglip-so400m-patch14-384",
]
LEGACY_LLM = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
]
NEXTGEN_LLM = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gpt-5.2-2025-12-11",
    "gpt-5-mini-2025-08-07",
]
ALL_MODELS = CONTRASTIVE + LEGACY_LLM + NEXTGEN_LLM

GENERATION_MAP = {}
for m in CONTRASTIVE:
    GENERATION_MAP[m] = "Contrastive"
for m in LEGACY_LLM:
    GENERATION_MAP[m] = "Legacy LLM"
for m in NEXTGEN_LLM:
    GENERATION_MAP[m] = "Next-Gen LLM"

MODEL_SHORT = {
    "clip-vit-base-patch32": "CLIP B/32",
    "clip-vit-base-patch16": "CLIP B/16",
    "clip-vit-large-patch14": "CLIP L/14",
    "siglip-base-patch16-512": "SigLIP B",
    "siglip-large-patch16-384": "SigLIP L",
    "siglip-so400m-patch14-384": "SigLIP SO",
    "gemini-2.5-flash-preview-05-20": "Gem2.5 Flash",
    "gemini-2.5-pro-preview-05-06": "Gem2.5 Pro",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "gemini-3-flash-preview": "Gem3 Flash",
    "gemini-3.1-pro-preview": "Gem3.1 Pro",
    "gpt-5.2-2025-12-11": "GPT-5.2",
    "gpt-5-mini-2025-08-07": "GPT-5 Mini",
}

SUCCESSOR_PAIRS = [
    ("gpt-4o-mini-2024-07-18", "gpt-5-mini-2025-08-07"),
    ("gpt-4o-2024-08-06", "gpt-5.2-2025-12-11"),
    ("gemini-2.5-flash-preview-05-20", "gemini-3-flash-preview"),
    ("gemini-2.5-pro-preview-05-06", "gemini-3.1-pro-preview"),
]

GEN_COLORS = {
    "Contrastive": "#7EB0D5",
    "Legacy LLM": "#E8927C",
    "Next-Gen LLM": "#5DA271",
}

GEN_ORDER = ["Contrastive", "Legacy LLM", "Next-Gen LLM"]


# ─── Data loading ────────────────────────────────────────────────────────────

def load_consistency_data():
    """Load consistency_data.json for every model/test and return a flat DataFrame."""
    records = []
    for model in ALL_MODELS:
        gen = GENERATION_MAP[model]
        short = MODEL_SHORT.get(model, model)
        for test in TESTS:
            path = CONSISTENCY_DIR / model / test / "consistency_data.json"
            if not path.exists():
                continue
            with open(path) as f:
                pairs = json.load(f)

            for pair in pairs:
                if len(pair) < 2:
                    continue
                p0, p1 = pair[0], pair[1]
                pred0, pred1 = p0.get("predicted"), p1.get("predicted")
                gt0, gt1 = p0.get("ground_truth"), p1.get("ground_truth")
                if pred0 is None or pred1 is None:
                    continue

                same_pred = int(pred0 == pred1)
                both_correct = int(pred0 == gt0 and pred1 == gt1)
                any_correct = int(pred0 == gt0 or pred1 == gt1)

                ds_key = " × ".join(sorted([p0.get("dataset", "?"),
                                             p1.get("dataset", "?")]))
                records.append({
                    "model": model,
                    "model_short": short,
                    "generation": gen,
                    "test": test,
                    "gt_class": gt0,
                    "dataset_pair": ds_key,
                    "same_pred": same_pred,
                    "both_correct": both_correct,
                    "any_correct": any_correct,
                    "pred0": pred0,
                    "pred1": pred1,
                })
    return pd.DataFrame(records)


# ─── Analysis 1: Generation-level consistency ────────────────────────────────

def plot_generation_comparison(df):
    """Bar + box plot comparing avg consistency by generation."""
    out = OUTPUT_DIR / "1_generation_comparison"
    out.mkdir(parents=True, exist_ok=True)

    # Per model+test consistency rate
    agg = (df.groupby(["model", "model_short", "generation", "test"])
             .agg(consistency=("same_pred", "mean"),
                  n_pairs=("same_pred", "count"))
             .reset_index())
    agg["consistency_pct"] = agg["consistency"] * 100

    # Average across tests per model
    model_avg = (agg.groupby(["model", "model_short", "generation"])
                    .agg(avg_consistency=("consistency_pct", "mean"),
                         std_consistency=("consistency_pct", "std"))
                    .reset_index())
    model_avg = model_avg.set_index("model").reindex(ALL_MODELS).reset_index()

    # ── Fig 1a: bar chart ordered by generation ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(model_avg))
    colors = [GEN_COLORS[g] for g in model_avg["generation"]]
    bars = ax.bar(x, model_avg["avg_consistency"], yerr=model_avg["std_consistency"],
                  color=colors, width=0.65, capsize=3, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, model_avg["avg_consistency"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_avg["model_short"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Avg Consistency (%)", fontsize=12)
    ax.set_title("Cross-Dataset Consistency by Model (avg ± std across 3 tests)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # generation dividers
    n_c = len(CONTRASTIVE)
    n_l = len(LEGACY_LLM)
    ax.axvline(n_c - 0.5, color="gray", ls=":", lw=1)
    ax.axvline(n_c + n_l - 0.5, color="gray", ls="--", lw=1.2)

    patches = [mpatches.Patch(color=GEN_COLORS[g], label=g) for g in GEN_ORDER]
    ax.legend(handles=patches, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "consistency_by_model.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Fig 1b: box plot by generation ──
    gen_agg = (agg.groupby(["model", "generation"])["consistency_pct"]
                  .mean().reset_index())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=gen_agg, x="generation", y="consistency_pct",
                order=GEN_ORDER, palette=GEN_COLORS, width=0.5, ax=ax)
    sns.stripplot(data=gen_agg, x="generation", y="consistency_pct",
                  order=GEN_ORDER, color="black", size=6, alpha=0.6, ax=ax)
    ax.set_ylabel("Avg Consistency (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Consistency Distribution by Generation", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "consistency_by_generation_box.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Generation means for text summary ──
    gen_means = gen_agg.groupby("generation")["consistency_pct"].agg(["mean", "std", "median"])
    gen_means.to_csv(out / "generation_summary.csv")
    print(f"  [1] Generation comparison → {out}")
    return agg, model_avg


# ─── Analysis 2: Cross-test variance ─────────────────────────────────────────

def plot_variance_analysis(agg):
    """Show that next-gen models have lower variance across the 3 tests."""
    out = OUTPUT_DIR / "2_variance_analysis"
    out.mkdir(parents=True, exist_ok=True)

    # Per-model variance across tests
    var_df = (agg.groupby(["model", "model_short", "generation"])["consistency_pct"]
                 .agg(["mean", "std", "min", "max"])
                 .reset_index())
    var_df["range"] = var_df["max"] - var_df["min"]
    var_df = var_df.set_index("model").reindex(ALL_MODELS).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cross-Test Stability: Why Next-Gen Models Are More Consistent",
                 fontsize=14, fontweight="bold")

    # Left: std dev per model
    ax = axes[0]
    x = np.arange(len(var_df))
    colors = [GEN_COLORS[GENERATION_MAP[m]] for m in var_df["model"]]
    bars = ax.bar(x, var_df["std"], color=colors, width=0.65)
    for bar, v in zip(bars, var_df["std"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(var_df["model_short"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Std Dev of Consistency (%)", fontsize=11)
    ax.set_title("Standard Deviation Across 3 Tests", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Right: range (max - min)
    ax = axes[1]
    bars = ax.bar(x, var_df["range"], color=colors, width=0.65)
    for bar, v in zip(bars, var_df["range"]):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(var_df["model_short"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Range (max − min) of Consistency (%)", fontsize=11)
    ax.set_title("Max−Min Range Across 3 Tests", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=GEN_COLORS[g], label=g) for g in GEN_ORDER]
    fig.legend(handles=patches, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "variance_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()

    var_df.to_csv(out / "variance_data.csv", index=False)

    # Generation-level variance summary
    gen_var = var_df.groupby("generation")[["std", "range"]].agg(["mean", "median"])
    gen_var.to_csv(out / "variance_by_generation.csv")
    print(f"  [2] Variance analysis → {out}")


# ─── Analysis 3: Per-class consistency breakdown ─────────────────────────────

def plot_perclass_consistency(df):
    """Which saint classes are hardest/easiest for consistency? How does gen help?"""
    out = OUTPUT_DIR / "3_perclass_breakdown"
    out.mkdir(parents=True, exist_ok=True)

    # Per generation × class consistency
    class_gen = (df.groupby(["generation", "gt_class"])
                   .agg(consistency=("same_pred", "mean"),
                        n_pairs=("same_pred", "count"))
                   .reset_index())
    class_gen["consistency_pct"] = class_gen["consistency"] * 100

    # Pivot: class × generation
    pivot = class_gen.pivot_table(index="gt_class", columns="generation",
                                  values="consistency_pct", aggfunc="mean")
    pivot = pivot.reindex(columns=GEN_ORDER)

    # Compute improvement: next-gen minus legacy
    if "Next-Gen LLM" in pivot.columns and "Legacy LLM" in pivot.columns:
        pivot["improvement"] = pivot["Next-Gen LLM"] - pivot["Legacy LLM"]
        pivot = pivot.sort_values("improvement", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
    pivot[GEN_ORDER].plot(kind="barh", ax=ax, color=[GEN_COLORS[g] for g in GEN_ORDER],
                          width=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Consistency (%)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Per-Class Consistency by Generation\n(sorted by Next-Gen improvement over Legacy)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "perclass_consistency.png", dpi=200, bbox_inches="tight")
    plt.close()

    pivot.to_csv(out / "perclass_data.csv")
    print(f"  [3] Per-class breakdown → {out}")


# ─── Analysis 4: Consistency through correctness ─────────────────────────────

def plot_correctness_driven_consistency(df):
    """Decompose consistency into: both correct, both wrong-same, divergent."""
    out = OUTPUT_DIR / "4_correctness_decomposition"
    out.mkdir(parents=True, exist_ok=True)

    def classify_pair(row):
        if row["same_pred"] == 1 and row["both_correct"] == 1:
            return "Both correct (consistent)"
        elif row["same_pred"] == 1 and row["both_correct"] == 0:
            return "Same wrong prediction"
        else:
            return "Different predictions"

    df = df.copy()
    df["pair_type"] = df.apply(classify_pair, axis=1)

    # Per model breakdown
    type_counts = (df.groupby(["model", "model_short", "generation", "pair_type"])
                     .size().reset_index(name="count"))
    totals = (df.groupby(["model", "model_short", "generation"])
                .size().reset_index(name="total"))
    type_counts = type_counts.merge(totals)
    type_counts["pct"] = type_counts["count"] / type_counts["total"] * 100

    # Pivot for stacked bar
    pivot = type_counts.pivot_table(index="model", columns="pair_type",
                                     values="pct", aggfunc="mean")
    ordered_cols = ["Both correct (consistent)", "Same wrong prediction",
                    "Different predictions"]
    pivot = pivot.reindex(columns=ordered_cols, fill_value=0)
    pivot = pivot.reindex(ALL_MODELS)

    short_labels = [MODEL_SHORT.get(m, m) for m in pivot.index]
    type_colors = ["#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(pivot))
    for col, color in zip(ordered_cols, type_colors):
        vals = pivot[col].values
        ax.bar(range(len(pivot)), vals, bottom=bottom, color=color,
               label=col, width=0.7, edgecolor="white", linewidth=0.5)
        # Add % labels in the middle of each segment
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white")
        bottom += vals

    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("% of Image Pairs", fontsize=12)
    ax.set_title("Decomposition of Consistency:\nNext-Gen models are consistent because they are correct, not just agreeing on wrong answers",
                 fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    n_c = len(CONTRASTIVE)
    n_l = len(LEGACY_LLM)
    ax.axvline(n_c - 0.5, color="gray", ls=":", lw=1)
    ax.axvline(n_c + n_l - 0.5, color="gray", ls="--", lw=1.2)

    plt.tight_layout()
    plt.savefig(out / "correctness_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Generation-level summary
    gen_summary = (df.groupby(["generation", "pair_type"])
                     .size().reset_index(name="count"))
    gen_totals = df.groupby("generation").size().reset_index(name="total")
    gen_summary = gen_summary.merge(gen_totals)
    gen_summary["pct"] = gen_summary["count"] / gen_summary["total"] * 100
    gen_summary.to_csv(out / "correctness_by_generation.csv", index=False)
    print(f"  [4] Correctness decomposition → {out}")


# ─── Analysis 5: Successor pair waterfall ─────────────────────────────────────

def plot_successor_waterfall(df):
    """For each legacy→next-gen pair, show consistency improvement waterfall."""
    out = OUTPUT_DIR / "5_successor_waterfall"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for old_m, new_m in SUCCESSOR_PAIRS:
        for test in TESTS:
            old_data = df[(df["model"] == old_m) & (df["test"] == test)]
            new_data = df[(df["model"] == new_m) & (df["test"] == test)]
            if old_data.empty or new_data.empty:
                continue
            old_cons = old_data["same_pred"].mean() * 100
            new_cons = new_data["same_pred"].mean() * 100
            old_both_correct = old_data["both_correct"].mean() * 100
            new_both_correct = new_data["both_correct"].mean() * 100
            records.append({
                "pair": f"{MODEL_SHORT[old_m]} → {MODEL_SHORT[new_m]}",
                "old_model": MODEL_SHORT[old_m],
                "new_model": MODEL_SHORT[new_m],
                "test": test,
                "old_consistency": old_cons,
                "new_consistency": new_cons,
                "delta": new_cons - old_cons,
                "old_both_correct": old_both_correct,
                "new_both_correct": new_both_correct,
                "delta_correct": new_both_correct - old_both_correct,
            })

    if not records:
        print("  [5] No successor pairs found, skipping")
        return

    rdf = pd.DataFrame(records)
    rdf.to_csv(out / "successor_data.csv", index=False)

    # Average across tests
    avg = rdf.groupby("pair")[["old_consistency", "new_consistency", "delta",
                                "old_both_correct", "new_both_correct", "delta_correct"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Successor Model Improvements (averaged across 3 tests)",
                 fontsize=14, fontweight="bold")

    # Left: consistency delta
    ax = axes[0]
    x = np.arange(len(avg))
    ax.bar(x - 0.2, avg["old_consistency"], 0.35, color=GEN_COLORS["Legacy LLM"],
           label="Legacy", edgecolor="white")
    ax.bar(x + 0.2, avg["new_consistency"], 0.35, color=GEN_COLORS["Next-Gen LLM"],
           label="Next-Gen", edgecolor="white")
    for i, (o, n, d) in enumerate(zip(avg["old_consistency"], avg["new_consistency"], avg["delta"])):
        ax.annotate(f"+{d:.1f}pp", xy=(i + 0.2, n + 1), fontsize=9,
                    ha="center", fontweight="bold", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(avg["pair"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Consistency (%)", fontsize=11)
    ax.set_title("Overall Consistency", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Right: both-correct delta
    ax = axes[1]
    ax.bar(x - 0.2, avg["old_both_correct"], 0.35, color=GEN_COLORS["Legacy LLM"],
           label="Legacy", edgecolor="white")
    ax.bar(x + 0.2, avg["new_both_correct"], 0.35, color=GEN_COLORS["Next-Gen LLM"],
           label="Next-Gen", edgecolor="white")
    for i, (o, n, d) in enumerate(zip(avg["old_both_correct"], avg["new_both_correct"], avg["delta_correct"])):
        ax.annotate(f"+{d:.1f}pp", xy=(i + 0.2, n + 1), fontsize=9,
                    ha="center", fontweight="bold", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(avg["pair"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Both Correct (%)", fontsize=11)
    ax.set_title("Both-Correct Consistency (quality-driven)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "successor_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [5] Successor waterfall → {out}")


# ─── Analysis 6: Statistical tests ───────────────────────────────────────────

def run_statistical_tests(df):
    """Mann-Whitney U tests between generation groups."""
    out = OUTPUT_DIR / "6_statistical_tests"
    out.mkdir(parents=True, exist_ok=True)

    # Per-model average consistency
    model_cons = (df.groupby(["model", "generation"])["same_pred"]
                    .mean().reset_index())
    model_cons["consistency_pct"] = model_cons["same_pred"] * 100

    results = []
    pairs_to_test = [
        ("Contrastive", "Legacy LLM"),
        ("Contrastive", "Next-Gen LLM"),
        ("Legacy LLM", "Next-Gen LLM"),
    ]
    for g1, g2 in pairs_to_test:
        vals1 = model_cons[model_cons["generation"] == g1]["consistency_pct"].values
        vals2 = model_cons[model_cons["generation"] == g2]["consistency_pct"].values
        if len(vals1) < 2 or len(vals2) < 2:
            continue
        stat, pval = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
        effect_size = stat / (len(vals1) * len(vals2))  # rank-biserial
        results.append({
            "group_1": g1,
            "group_2": g2,
            "n_1": len(vals1),
            "n_2": len(vals2),
            "mean_1": np.mean(vals1),
            "mean_2": np.mean(vals2),
            "U_statistic": stat,
            "p_value": pval,
            "significant_0.05": pval < 0.05,
            "effect_size_r": effect_size,
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv(out / "mannwhitney_results.csv", index=False)

    # Pretty print
    summary_lines = ["# Statistical Significance Tests (Mann-Whitney U)", ""]
    summary_lines.append("Null hypothesis: no difference in consistency between groups.\n")
    for _, row in rdf.iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "n.s."
        summary_lines.append(
            f"- {row['group_1']} (mean={row['mean_1']:.1f}%) vs "
            f"{row['group_2']} (mean={row['mean_2']:.1f}%): "
            f"U={row['U_statistic']:.0f}, p={row['p_value']:.4f} {sig}"
        )
    summary_lines.append("")

    with open(out / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"  [6] Statistical tests → {out}")
    for line in summary_lines[2:]:
        if line.strip():
            print(f"      {line}")


# ─── Analysis 7: Dataset-pair specific trends ────────────────────────────────

def plot_dataset_pair_trends(df):
    """How does consistency vary by dataset pair across generations?"""
    out = OUTPUT_DIR / "7_dataset_pair_trends"
    out.mkdir(parents=True, exist_ok=True)

    ds_gen = (df.groupby(["generation", "dataset_pair"])
                .agg(consistency=("same_pred", "mean"),
                     n_pairs=("same_pred", "count"))
                .reset_index())
    ds_gen["consistency_pct"] = ds_gen["consistency"] * 100

    pivot = ds_gen.pivot_table(index="dataset_pair", columns="generation",
                                values="consistency_pct")
    pivot = pivot.reindex(columns=GEN_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=[GEN_COLORS[g] for g in GEN_ORDER],
               width=0.75, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Consistency (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Consistency by Dataset Pair and Generation", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "dataset_pair_trends.png", dpi=200, bbox_inches="tight")
    plt.close()

    pivot.to_csv(out / "dataset_pair_data.csv")
    print(f"  [7] Dataset-pair trends → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading consistency data...")
    df = load_consistency_data()

    if df.empty:
        print("No consistency data found. Run generate_consistency_data.py first.")
        return

    print(f"Loaded {len(df)} pair records across {df['model'].nunique()} models.\n")

    agg, model_avg = plot_generation_comparison(df)
    plot_variance_analysis(agg)
    plot_perclass_consistency(df)
    plot_correctness_driven_consistency(df)
    plot_successor_waterfall(df)
    run_statistical_tests(df)
    plot_dataset_pair_trends(df)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
