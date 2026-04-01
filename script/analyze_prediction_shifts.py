#!/usr/bin/env python3
"""
Analyze prediction shifts between old and new LLM generations across all datasets.

Three analyses:
  1. Error Transition Matrix  – correct→correct / correct→wrong / wrong→correct / wrong→wrong
                                + class-level prediction change heatmap per model pair
  4. Confidence Shift         – max-prob (confidence), entropy, top-2 margin from probs.npy
  6. Cluster Convergence      – near-duplicate pair agreement rate (from consistency_data.json)

Outputs saved to:  dataset/shift_analysis/
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

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
OUTPUT_DIR      = BASE_DIR / "dataset" / "shift_analysis"
CONSISTENCY_DIR = BASE_DIR / "dataset" / "consistency"

DATASETS = ["ArtDL", "ICONCLASS", "wikidata"]
TESTS    = ["test_1", "test_2", "test_3"]

# ─── Model definitions ────────────────────────────────────────────────────────
OLD_LLM = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]
NEW_LLM = [
    "gpt-5-mini-2025-08-07",
    "gpt-5.2-2025-12-11",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]
BASELINE = [
    "clip-vit-base-patch32",
    "clip-vit-base-patch16",
    "clip-vit-large-patch14",
    "siglip-base-patch16-512",
    "siglip-large-patch16-384",
    "siglip-so400m-patch14-384",
]
ALL_MODELS = OLD_LLM + NEW_LLM + BASELINE

# Natural successor pairs for idea-1 pairwise transition analysis
MODEL_PAIRS = [
    ("gpt-4o-mini-2024-07-18",  "gpt-5-mini-2025-08-07"),
    ("gpt-4o-2024-08-06",       "gpt-5.2-2025-12-11"),
    ("gemini-2.5-flash-lite",   "gemini-3.1-flash-lite-preview"),
    ("gemini-2.5-flash",        "gemini-3-flash-preview"),
    ("gemini-2.5-pro",          "gemini-3.1-pro-preview"),
]

MODEL_LABELS = {
    "gpt-4o-mini-2024-07-18":        "GPT-4o Mini",
    "gpt-4o-2024-08-06":             "GPT-4o (Aug)",
    "gpt-4o-2024-11-20":             "GPT-4o (Nov)",
    "gemini-2.5-flash-lite":         "Gem2.5 FL",
    "gemini-2.5-flash":              "Gem2.5 Flash",
    "gemini-2.5-pro":                "Gem2.5 Pro",
    "gpt-5-mini-2025-08-07":         "GPT-5 Mini",
    "gpt-5.2-2025-12-11":            "GPT-5.2",
    "gemini-3.1-flash-lite-preview": "Gem3.1 FL",
    "gemini-3-flash-preview":        "Gem3 Flash",
    "gemini-3.1-pro-preview":        "Gem3.1 Pro",
    "clip-vit-base-patch32":         "CLIP B/32",
    "clip-vit-base-patch16":         "CLIP B/16",
    "clip-vit-large-patch14":        "CLIP L/14",
    "siglip-base-patch16-512":       "SigLIP B",
    "siglip-large-patch16-384":      "SigLIP L",
    "siglip-so400m-patch14-384":     "SigLIP SO",
}

GEN_COLORS = {
    "old":      "#E8927C",
    "new":      "#5DA271",
    "baseline": "#7EB0D5",
}

def gen_of(model):
    if model in OLD_LLM:     return "old"
    if model in NEW_LLM:     return "new"
    return "baseline"

# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_classes(dataset: str, test: str):
    path = BASE_DIR / "dataset" / f"{dataset}-data" / "classes.csv"
    df = pd.read_csv(path)
    col = "Label" if test in ("test_1", "test_3") else "Description"
    return list(df[["ID", col]].itertuples(index=False, name=None))


def load_ground_truth(dataset: str) -> dict:
    path = BASE_DIR / "dataset" / f"{dataset}-data" / "2_ground_truth.json"
    with open(path) as f:
        data = json.load(f)
    return {item["item"]: item["class"] for item in data}


def load_test_images(dataset: str) -> list:
    path = BASE_DIR / "dataset" / f"{dataset}-data" / "2_test.txt"
    return path.read_text().splitlines()


def load_predictions(model: str, dataset: str, test: str,
                     test_images: list, classes: list) -> dict:
    """Return {image_name: (predicted_class_id, probs_array)}, or {} if missing."""
    model_dir  = BASE_DIR / test / dataset / model
    probs_path = model_dir / "probs.npy"
    if not probs_path.exists():
        return {}

    probs = np.load(probs_path)
    ids_path = model_dir / "image_ids.txt"
    images = ids_path.read_text().splitlines() if ids_path.exists() else test_images

    class_ids = [c[0] for c in classes]
    result = {}
    for i, img in enumerate(images[: len(probs)]):
        pred_idx = int(probs[i].argmax())
        pred_cls = class_ids[pred_idx] if pred_idx < len(class_ids) else None
        result[img] = (pred_cls, probs[i].astype(float))
    return result


# ─── Idea 1: Error Transition Matrix ─────────────────────────────────────────

def analyze_transitions(all_preds: dict, ground_truth: dict, classes: list,
                        dataset: str, test: str):
    out = OUTPUT_DIR / "idea1_transitions" / dataset / test
    out.mkdir(parents=True, exist_ok=True)
    class_ids = [c[0] for c in classes]

    # ── 2×2 transition rates per successor pair ──
    pair_records = []
    for old_m, new_m in MODEL_PAIRS:
        old_p = all_preds.get(old_m, {})
        new_p = all_preds.get(new_m, {})
        common = [img for img in set(old_p) & set(new_p) if ground_truth.get(img)]
        if not common:
            continue

        counts = {"correct→correct": 0, "correct→wrong": 0,
                  "wrong→correct":   0, "wrong→wrong":   0}
        for img in common:
            gt       = ground_truth[img]
            old_ok   = old_p[img][0] == gt
            new_ok   = new_p[img][0] == gt
            key      = ("correct" if old_ok else "wrong") + "→" + ("correct" if new_ok else "wrong")
            counts[key] += 1

        total = sum(counts.values())
        pair_records.append({
            "old_model": MODEL_LABELS[old_m],
            "new_model": MODEL_LABELS[new_m],
            "pair":      f"{MODEL_LABELS[old_m]} → {MODEL_LABELS[new_m]}",
            "n_images":  total,
            **{k: round(v / total * 100, 1) for k, v in counts.items()},
            **{f"{k}_n": v for k, v in counts.items()},
        })

        # ── per-class transition heatmap ──
        cm = np.zeros((len(class_ids), len(class_ids)), dtype=int)
        for img in common:
            op = old_p[img][0]
            np_ = new_p[img][0]
            if op in class_ids and np_ in class_ids:
                cm[class_ids.index(op), class_ids.index(np_)] += 1

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=class_ids, yticklabels=class_ids,
                    ax=ax, cbar_kws={"label": "% of old-model row"})
        ax.set_xlabel(f"New model ({MODEL_LABELS[new_m]}) prediction", fontsize=11)
        ax.set_ylabel(f"Old model ({MODEL_LABELS[old_m]}) prediction", fontsize=11)
        ax.set_title(
            f"Class Transition: {MODEL_LABELS[old_m]} → {MODEL_LABELS[new_m]}\n"
            f"{dataset} / {test}  (row-normalised %)", fontsize=12)
        plt.tight_layout()
        safe = f"{old_m}_to_{new_m}"
        plt.savefig(out / f"class_heatmap_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()

    if not pair_records:
        print(f"  [Idea 1] {dataset}/{test}: no model pairs found, skipping")
        return

    df = pd.DataFrame(pair_records)
    df.to_csv(out / "transition_rates.csv", index=False)

    # ── grouped bar chart of transition rates ──
    cats   = ["correct→correct", "correct→wrong", "wrong→correct", "wrong→wrong"]
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
    labels = ["Correct → Correct", "Correct → Wrong (regression)",
              "Wrong → Correct (improvement)", "Wrong → Wrong"]

    x = np.arange(len(pair_records))
    w = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(pair_records) * 2.5), 6))
    for i, (cat, col, lab) in enumerate(zip(cats, colors, labels)):
        vals = df[cat].tolist()
        bars = ax.bar(x + i * w, vals, w, label=lab, color=col)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([r["pair"] for r in pair_records],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% of images", fontsize=11)
    ax.set_title(f"Error Transitions (old → new generation)  |  {dataset} / {test}", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out / "transition_bars.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [Idea 1] {dataset}/{test} → {out}")


# ─── Idea 1b: Regression Detail Analysis ─────────────────────────────────────

def analyze_regression_details(all_preds: dict, ground_truth: dict, classes: list,
                                dataset: str, test: str):
    """
    For each model pair, drill into correct→wrong cases:
      - which true classes regressed most?
      - what did the new model predict instead? (confusion target)
      - net change (improvements − regressions) per class
      - short label using first word(s) of class name for readability
    """
    out = OUTPUT_DIR / "idea1_transitions" / dataset / test
    out.mkdir(parents=True, exist_ok=True)

    class_ids    = [c[0]  for c in classes]
    class_labels = {c[0]: c[1] for c in classes}

    def short(cid):
        """Return a readable short label for a class ID."""
        lbl = class_labels.get(cid, cid)
        # Use first two words of the label, stripping leading 'St.'
        words = lbl.replace("St. ", "").split()
        return " ".join(words[:2])

    for old_m, new_m in MODEL_PAIRS:
        old_p = all_preds.get(old_m, {})
        new_p = all_preds.get(new_m, {})
        common = [img for img in set(old_p) & set(new_p) if ground_truth.get(img)]
        if not common:
            continue

        regressions, improvements = [], []
        for img in common:
            gt     = ground_truth[img]
            old_ok = old_p[img][0] == gt
            new_ok = new_p[img][0] == gt
            if old_ok and not new_ok:
                regressions.append({"img": img, "true": gt,
                                    "new_pred": new_p[img][0]})
            elif not old_ok and new_ok:
                improvements.append({"img": img, "true": gt,
                                     "old_pred": old_p[img][0]})

        if not regressions:
            continue

        reg_df = pd.DataFrame(regressions)
        imp_df = pd.DataFrame(improvements)
        safe   = f"{old_m}_to_{new_m}"
        label  = f"{MODEL_LABELS[old_m]} → {MODEL_LABELS[new_m]}"

        # ── Save per-image regression CSV ──
        reg_df.to_csv(out / f"regression_cases_{safe}.csv", index=False)

        # ── Per-class regression count + confusion target breakdown ──
        reg_by_class = (reg_df.groupby(["true", "new_pred"])
                               .size()
                               .reset_index(name="n")
                               .sort_values(["true", "n"], ascending=[True, False]))
        reg_by_class.to_csv(out / f"regression_by_class_{safe}.csv", index=False)

        # Pivot: true class × new_pred
        pivot = reg_by_class.pivot(index="true", columns="new_pred", values="n").fillna(0).astype(int)
        # Reindex to full class list order, keep only classes with ≥1 regression
        active_true = [c for c in class_ids if c in pivot.index]
        active_pred = [c for c in class_ids if c in pivot.columns]
        pivot = pivot.reindex(index=active_true, columns=active_pred, fill_value=0)

        short_true = [short(c) for c in active_true]
        short_pred = [short(c) for c in active_pred]

        # ── Heatmap: regression confusion (true class → new model's wrong prediction) ──
        fig, ax = plt.subplots(figsize=(max(8, len(active_pred) * 0.9),
                                        max(6, len(active_true) * 0.7)))
        sns.heatmap(pivot.values, annot=True, fmt="d", cmap="Reds",
                    xticklabels=short_pred, yticklabels=short_true,
                    ax=ax, linewidths=0.4, linecolor="#ddd",
                    cbar_kws={"label": "# of regressions"})
        ax.set_xlabel(f"New model ({MODEL_LABELS[new_m]}) wrong prediction", fontsize=10)
        ax.set_ylabel("True class (was correct for old model)", fontsize=10)
        ax.set_title(f"Regression Confusion: {label}\n{dataset} / {test}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out / f"regression_heatmap_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Net-change bar chart: improvements − regressions per true class ──
        reg_counts = reg_df["true"].value_counts().reindex(class_ids, fill_value=0)
        imp_counts = pd.Series(0, index=class_ids)
        if not imp_df.empty:
            imp_counts = imp_df["true"].value_counts().reindex(class_ids, fill_value=0)

        net = imp_counts - reg_counts
        # Keep only classes with nonzero net or nonzero regression
        mask = (reg_counts > 0) | (imp_counts > 0)
        net_active    = net[mask]
        reg_active    = reg_counts[mask]
        imp_active    = imp_counts[mask]
        active_cids   = net_active.index.tolist()
        active_shorts = [short(c) for c in active_cids]

        fig, axes = plt.subplots(1, 2, figsize=(max(14, len(active_cids) * 1.1), 6),
                                 gridspec_kw={"width_ratios": [2, 1]})
        fig.suptitle(f"Per-class Regression/Improvement: {label}\n{dataset} / {test}",
                     fontsize=13, fontweight="bold")

        # Left: stacked bar (regressions in red, improvements in green)
        ax = axes[0]
        x = np.arange(len(active_cids))
        ax.bar(x, imp_active.values, color="#4CAF50", label="Improvements (wrong→correct)", width=0.6)
        ax.bar(x, -reg_active.values, color="#F44336", label="Regressions (correct→wrong)", width=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        for i, (imp, reg) in enumerate(zip(imp_active.values, reg_active.values)):
            if imp > 0:
                ax.text(i, imp + 0.2, str(imp), ha="center", fontsize=8, color="#2E7D32")
            if reg > 0:
                ax.text(i, -reg - 0.5, str(reg), ha="center", fontsize=8, color="#B71C1C")
        ax.set_xticks(x)
        ax.set_xticklabels(active_shorts, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Image count", fontsize=11)
        ax.set_title("Regressions (−) and Improvements (+) per class", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Right: net change, sorted
        ax2 = axes[1]
        net_sorted   = net_active.sort_values()
        short_sorted = [short(c) for c in net_sorted.index]
        colors_net   = ["#F44336" if v < 0 else "#4CAF50" for v in net_sorted.values]
        bars = ax2.barh(range(len(net_sorted)), net_sorted.values, color=colors_net, height=0.65)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_yticks(range(len(net_sorted)))
        ax2.set_yticklabels(short_sorted, fontsize=9)
        for bar, v in zip(bars, net_sorted.values):
            ax2.text(v + (0.1 if v >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
                     f"{v:+d}", ha="left" if v >= 0 else "right", va="center", fontsize=8)
        ax2.set_xlabel("Net change (improvements − regressions)", fontsize=10)
        ax2.set_title("Net class change", fontsize=11)
        ax2.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out / f"regression_net_change_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Top regression patterns as a ranked list (for quick reading) ──
        top = (reg_by_class.head(20)
                            .assign(true_short =lambda d: d["true"].map(short),
                                    pred_short =lambda d: d["new_pred"].map(short)))
        top.to_csv(out / f"regression_top_patterns_{safe}.csv", index=False)

    print(f"  [Idea 1b] {dataset}/{test} regression details → {out}")


# ─── Idea 4: Confidence Shift Analysis ───────────────────────────────────────

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def analyze_confidence(all_preds: dict, ground_truth: dict,
                       dataset: str, test: str):
    out = OUTPUT_DIR / "idea4_confidence" / dataset / test
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for model in ALL_MODELS:
        preds = all_preds.get(model, {})
        if not preds:
            continue
        gen = gen_of(model)
        for img, (pred_cls, probs_vec) in preds.items():
            sorted_p = np.sort(probs_vec)[::-1]
            records.append({
                "model":      MODEL_LABELS[model],
                "model_id":   model,
                "generation": gen,
                "confidence": float(sorted_p[0]),
                "entropy":    _entropy(probs_vec),
                "margin":     float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0]),
                "correct":    int(pred_cls == ground_truth.get(img)),
            })

    if not records:
        return

    df = pd.DataFrame(records)
    df.to_csv(out / "confidence_data.csv", index=False)

    # Ordered model axis: baseline → old → new
    model_order = [MODEL_LABELS[m] for m in BASELINE + OLD_LLM + NEW_LLM
                   if MODEL_LABELS[m] in df["model"].unique()]
    palette = {MODEL_LABELS[m]: GEN_COLORS[gen_of(m)] for m in ALL_MODELS}

    metrics = ["confidence", "entropy", "margin"]
    ylabels = ["Max Probability (confidence)", "Entropy (bits)", "Top-1 − Top-2 Margin"]

    # ── box plots ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Confidence Shift  |  {dataset} / {test}", fontsize=14, fontweight="bold")

    n_baseline = sum(1 for m in BASELINE if MODEL_LABELS[m] in df["model"].unique())
    n_old      = sum(1 for m in OLD_LLM   if MODEL_LABELS[m] in df["model"].unique())

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        df["_gen"] = df["model"].map({MODEL_LABELS[m]: gen_of(m) for m in ALL_MODELS})
        sns.boxplot(data=df, x="model", y=metric, order=model_order,
                    hue="_gen", hue_order=["baseline", "old", "new"],
                    palette=GEN_COLORS, legend=False,
                    ax=ax, width=0.55, fliersize=1.5, linewidth=0.8)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # generation dividers
        if n_baseline > 0:
            ax.axvline(n_baseline - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.axvline(n_baseline + n_old - 0.5, color="gray", linestyle="--", linewidth=1.2)

    # generation legend
    patches = [mpatches.Patch(color=GEN_COLORS[g], label=g.capitalize()) for g in ("baseline", "old", "new")]
    fig.legend(handles=patches, loc="upper right", fontsize=10, title="Generation")
    plt.tight_layout()
    plt.savefig(out / "confidence_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── confidence distribution: correct vs wrong, old gen vs new gen ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Confidence by Correctness (LLM only)  |  {dataset} / {test}", fontsize=13)

    llm_df = df[df["generation"].isin(["old", "new"])]
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for gen, color in [("old", GEN_COLORS["old"]), ("new", GEN_COLORS["new"])]:
            for correct, ls, suffix in [(1, "-", "✓ correct"), (0, "--", "✗ wrong")]:
                vals = llm_df[(llm_df["generation"] == gen) &
                              (llm_df["correct"]    == correct)][metric].dropna()
                if len(vals) > 0:
                    ax.hist(vals, bins=30, alpha=0.35, color=color, linestyle=ls,
                            label=f"{'Old' if gen == 'old' else 'New'} {suffix}",
                            density=True, histtype="stepfilled",
                            edgecolor=color, linewidth=0.8)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "confidence_by_correctness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── summary table ──
    summary = (df.groupby(["generation", "model"])[metrics]
                 .agg(["mean", "median", "std"])
                 .round(4))
    summary.to_csv(out / "confidence_summary.csv")

    print(f"  [Idea 4] {dataset}/{test} → {out}")


# ─── Idea 6: Perceptual Hash Cluster Analysis ─────────────────────────────────

def analyze_clusters():
    out = OUTPUT_DIR / "idea6_clusters"
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for model in ALL_MODELS:
        gen = gen_of(model)
        for test in TESTS:
            jp = CONSISTENCY_DIR / model / test / "consistency_data.json"
            if not jp.exists():
                continue
            with open(jp) as f:
                pairs = json.load(f)

            ds_counts = defaultdict(lambda: {"same": 0, "total": 0,
                                             "same_correct": 0, "diff_correct": 0})
            for pair in pairs:
                if len(pair) < 2:
                    continue
                p0, p1 = pair[0], pair[1]
                pred0, pred1 = p0.get("predicted"), p1.get("predicted")
                gt0,   gt1   = p0.get("ground_truth"), p1.get("ground_truth")
                if pred0 is None or pred1 is None:
                    continue

                ds0 = p0.get("dataset", "?")
                ds1 = p1.get("dataset", "?")
                # canonical order so ArtDL×ICONCLASS == ICONCLASS×ArtDL
                ds_key = "×".join(sorted([ds0, ds1]))

                ds_counts[ds_key]["total"] += 1
                if pred0 == pred1:
                    ds_counts[ds_key]["same"] += 1
                    # convergence is "good" only if converging on correct class
                    if gt0 and pred0 == gt0:
                        ds_counts[ds_key]["same_correct"] += 1
                else:
                    # divergence but both correct is still OK
                    if gt0 and gt1 and pred0 == gt0 and pred1 == gt1:
                        ds_counts[ds_key]["diff_correct"] += 1

            for ds_key, c in ds_counts.items():
                total = c["total"]
                records.append({
                    "model":          MODEL_LABELS[model],
                    "model_id":       model,
                    "generation":     gen,
                    "test":           test,
                    "dataset_pair":   ds_key,
                    "total_pairs":    total,
                    "same_pred":      c["same"],
                    "same_correct":   c["same_correct"],
                    "diff_correct":   c["diff_correct"],
                    "convergence_%":  c["same"] / total * 100 if total else 0,
                    "correct_convergence_%": c["same_correct"] / total * 100 if total else 0,
                })

    if not records:
        print("  [Idea 6] No consistency data found")
        return

    df = pd.DataFrame(records)
    df.to_csv(out / "cluster_convergence.csv", index=False)

    # Ordered model axis: baseline → old → new
    model_order_ids = [m for m in BASELINE + OLD_LLM + NEW_LLM
                       if m in df["model_id"].unique()]
    model_order_lbl = [MODEL_LABELS[m] for m in model_order_ids]
    palette = {MODEL_LABELS[m]: GEN_COLORS[gen_of(m)] for m in model_order_ids}

    n_baseline = sum(1 for m in BASELINE  if m in df["model_id"].unique())
    n_old      = sum(1 for m in OLD_LLM   if m in df["model_id"].unique())

    def _add_dividers(ax):
        if n_baseline > 0:
            ax.axvline(n_baseline - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.axvline(n_baseline + n_old - 0.5, color="gray", linestyle="--", linewidth=1.2)

    # ── overall convergence (mean across tests + dataset pairs) ──
    agg = (df.groupby(["model_id", "model", "generation"])
             [["convergence_%", "correct_convergence_%"]]
             .mean()
             .reset_index())
    agg_ord = (agg.set_index("model_id")
                  .reindex(model_order_ids)
                  .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(model_order_ids) * 1.4), 6))
    fig.suptitle("Cluster Convergence: Near-Duplicate Pair Agreement", fontsize=14, fontweight="bold")

    for ax, col, title in zip(axes,
                               ["convergence_%", "correct_convergence_%"],
                               ["Any-class Convergence", "Correct-class Convergence"]):
        colors = [GEN_COLORS[gen_of(mid)] for mid in agg_ord["model_id"]]
        bars = ax.bar(agg_ord["model"], agg_ord[col], color=colors, width=0.65)
        for bar, v in zip(bars, agg_ord[col]):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", fontsize=7)
        ax.set_xticks(range(len(agg_ord)))
        ax.set_xticklabels(agg_ord["model"].tolist(), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Avg %", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        _add_dividers(ax)

    patches = [mpatches.Patch(color=GEN_COLORS[g], label=g.capitalize())
               for g in ("baseline", "old", "new")]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "convergence_overall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── per-test breakdown ──
    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(model_order_ids) * 2), 6))
    fig.suptitle("Cluster Convergence per Test Run", fontsize=13, fontweight="bold")
    for ax, test in zip(axes, TESTS):
        sub = (df[df["test"] == test]
               .groupby(["model_id", "model", "generation"])["convergence_%"]
               .mean().reset_index())
        sub_ord = sub.set_index("model_id").reindex(model_order_ids).reset_index()
        colors = [GEN_COLORS[gen_of(mid)] for mid in sub_ord["model_id"]]
        vals = sub_ord["convergence_%"].tolist()
        bars = ax.bar(sub_ord["model"], vals, color=colors, width=0.65)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", fontsize=7)
        ax.set_title(test, fontsize=11)
        ax.set_xticks(range(len(sub_ord)))
        ax.set_xticklabels(sub_ord["model"].tolist(), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Convergence (%)")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        _add_dividers(ax)
    plt.tight_layout()
    plt.savefig(out / "convergence_per_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── per dataset-pair breakdown ──
    for ds_pair in sorted(df["dataset_pair"].unique()):
        sub = (df[df["dataset_pair"] == ds_pair]
               .groupby(["model_id", "model", "generation"])["convergence_%"]
               .mean().reset_index())
        sub_ord = sub.set_index("model_id").reindex(model_order_ids).reset_index()

        fig, ax = plt.subplots(figsize=(max(10, len(model_order_ids) * 1.2), 5))
        colors = [GEN_COLORS[gen_of(mid)] for mid in sub_ord["model_id"]]
        vals = sub_ord["convergence_%"].tolist()
        bars = ax.bar(sub_ord["model"], vals, color=colors, width=0.65)
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_title(f"Cluster Convergence — {ds_pair} pairs", fontsize=11)
        ax.set_xticks(range(len(sub_ord)))
        ax.set_xticklabels(sub_ord["model"].tolist(), rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Convergence (%)")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        _add_dividers(ax)

        patches = [mpatches.Patch(color=GEN_COLORS[g], label=g.capitalize())
                   for g in ("baseline", "old", "new")]
        ax.legend(handles=patches, fontsize=9)
        plt.tight_layout()
        safe = ds_pair.replace("×", "_x_")
        plt.savefig(out / f"convergence_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [Idea 6] → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        data_dir = BASE_DIR / "dataset" / f"{dataset}-data"
        if not data_dir.exists():
            print(f"[SKIP] {dataset}: data dir not found")
            continue

        ground_truth = load_ground_truth(dataset)
        test_images  = load_test_images(dataset)

        for test in TESTS:
            print(f"\n── {dataset} / {test} ──")
            classes = load_classes(dataset, test)

            all_preds: dict = {}
            for model in ALL_MODELS:
                preds = load_predictions(model, dataset, test, test_images, classes)
                if preds:
                    all_preds[model] = preds
                else:
                    print(f"    [SKIP] {model}")

            if not all_preds:
                continue

            analyze_transitions(all_preds, ground_truth, classes, dataset, test)
            analyze_regression_details(all_preds, ground_truth, classes, dataset, test)
            analyze_confidence(all_preds, ground_truth, dataset, test)

    # Idea 6 runs once across all datasets (uses cross-dataset pair data)
    print("\n── Idea 6: Cluster Convergence ──")
    analyze_clusters()

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
