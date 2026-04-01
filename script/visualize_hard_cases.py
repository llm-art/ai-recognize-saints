#!/usr/bin/env python3
"""
Visualize the hardest consistency cases across all models.

For each model, compare predictions for the same image across test_1/test_2/test_3.
An image is "unstable" in a model if it receives different predictions across the 3 runs.
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CONSISTENCY_DIR = Path(__file__).parent.parent / "dataset" / "consistency"
OUTPUT_PATH = CONSISTENCY_DIR / "hard_cases_visualization.png"

MODEL_GROUPS = {
    "GPT": ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "gpt-5-mini-2025-08-07", "gpt-5.2-2025-12-11"],
    "Gemini": ["gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-preview-05-20",
               "gemini-2.5-pro-preview-05-06", "gemini-3-flash-preview", "gemini-3.1-pro-preview"],
    "CLIP": ["clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14"],
    "SigLIP": ["siglip-base-patch16-512", "siglip-large-patch16-384", "siglip-so400m-patch14-384"],
}

SAINT_LABELS = {
    "11H(JEROME)": "Jerome",
    "11H(PETER)": "Peter",
    "11H(JOHN THE BAPTIST)": "John the Baptist",
    "11H(FRANCIS)": "Francis",
    "11HH(MARY MAGDALENE)": "Mary Magdalene",
}

GROUP_COLORS = {"GPT": "#4C72B0", "Gemini": "#55A868", "CLIP": "#C44E52", "SigLIP": "#DD8452"}


def load_model_data(model_name):
    """Load test_1, test_2, test_3 data for a model. Returns dict: test -> list of {name, dataset, gt, pred}."""
    model_dir = CONSISTENCY_DIR / model_name
    tests = {}
    for test in ["test_1", "test_2", "test_3"]:
        json_path = model_dir / test / "consistency_data.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            pairs = json.load(f)
        images = {}
        for pair in pairs:
            for img in pair:
                key = (img["name"], img["dataset"])
                images[key] = img.get("predicted")
        tests[test] = images
    return tests


def image_stability(tests):
    """
    For each image seen in at least 2 tests, return True if all predictions agree.
    Returns dict: (name, dataset) -> (is_stable: bool, gt, predictions_tuple)
    """
    # Gather ground truths from any test
    gt_map = {}
    all_keys = set()
    for test_data in tests.values():
        all_keys.update(test_data.keys())

    # Load ground truths from json (any test)
    for model_tests in [tests]:
        for test_data in model_tests.values():
            pass  # just need keys

    result = {}
    for key in all_keys:
        preds = [tests[t].get(key) for t in sorted(tests.keys()) if key in tests[t]]
        preds = [p for p in preds if p is not None]
        if len(preds) < 2:
            continue
        is_stable = len(set(preds)) == 1
        result[key] = (is_stable, preds)
    return result


def extract_saint(label):
    """Return clean saint name from ICONCLASS label."""
    if label is None:
        return "Unknown"
    return SAINT_LABELS.get(label, label)


def main():
    # --- Step 1: Collect per-image stability across all models ---
    # image_instability[(name, dataset)] = count of models where this image was unstable
    image_instability = defaultdict(int)
    image_gt = {}
    image_total_models = defaultdict(int)

    # Also collect per-saint instability per model group
    saint_group_instability = defaultdict(lambda: defaultdict(list))  # group -> saint -> [stable bools]

    all_models = [m for models in MODEL_GROUPS.values() for m in models]

    for group, models in MODEL_GROUPS.items():
        for model in models:
            tests = load_model_data(model)
            if len(tests) < 2:
                continue
            stability = image_stability(tests)

            # Also load GT from json
            model_dir = CONSISTENCY_DIR / model
            for test in ["test_1", "test_2", "test_3"]:
                json_path = model_dir / test / "consistency_data.json"
                if not json_path.exists():
                    continue
                with open(json_path) as f:
                    pairs = json.load(f)
                for pair in pairs:
                    for img in pair:
                        key = (img["name"], img["dataset"])
                        if img.get("ground_truth"):
                            image_gt[key] = img["ground_truth"]
                break

            for key, (is_stable, preds) in stability.items():
                image_total_models[key] += 1
                if not is_stable:
                    image_instability[key] += 1
                gt = image_gt.get(key)
                saint = extract_saint(gt)
                saint_group_instability[group][saint].append(is_stable)

    # --- Step 2: Per-saint instability rate per group ---
    saints = list(SAINT_LABELS.values())
    groups = list(MODEL_GROUPS.keys())

    saint_rates = np.zeros((len(groups), len(saints)))
    for gi, group in enumerate(groups):
        for si, saint in enumerate(saints):
            vals = saint_group_instability[group].get(saint, [])
            if vals:
                instability_rate = 1 - (sum(vals) / len(vals))
                saint_rates[gi, si] = instability_rate * 100

    # --- Step 3: Top N most unstable images ---
    # instability score: fraction of models where the image is unstable
    scored = []
    for key, count in image_instability.items():
        total = image_total_models[key]
        if total >= 3:  # only include images tested in at least 3 models
            score = count / total
            gt = image_gt.get(key, "")
            saint = extract_saint(gt)
            scored.append((score, count, total, key[0], key[1], saint))

    scored.sort(reverse=True)
    top_n = scored[:15]

    # --- Plot ---
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#F8F8F8")

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax_heatmap = fig.add_subplot(gs[0, :])   # top: heatmap spanning both columns
    ax_bar     = fig.add_subplot(gs[1, 0])   # bottom-left: top unstable images
    ax_pie     = fig.add_subplot(gs[1, 1])   # bottom-right: saint distribution of unstable images

    fig.suptitle("Hardest Consistency Cases Across Models", fontsize=17, fontweight="bold", y=0.97)

    # ---- Heatmap: per-group, per-saint instability ----
    im = ax_heatmap.imshow(saint_rates, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax_heatmap.set_xticks(range(len(saints)))
    ax_heatmap.set_xticklabels(saints, fontsize=11)
    ax_heatmap.set_yticks(range(len(groups)))
    ax_heatmap.set_yticklabels(groups, fontsize=11)
    ax_heatmap.set_title("Instability Rate (%) per Saint × Model Group", fontsize=12, pad=8)
    for gi in range(len(groups)):
        for si in range(len(saints)):
            val = saint_rates[gi, si]
            color = "white" if val > 60 else "black"
            ax_heatmap.text(si, gi, f"{val:.0f}%", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.025, pad=0.02)
    cbar.set_label("Instability %", fontsize=9)

    # ---- Bar chart: top 15 most unstable images ----
    if top_n:
        labels = [f"{row[3][:18]}\n({row[4]}, {row[5]})" for row in top_n]
        scores = [row[0] * 100 for row in top_n]
        saint_names = [row[5] for row in top_n]
        saint_color_map = {s: c for s, c in zip(saints, plt.cm.tab10.colors)}
        bar_colors = [saint_color_map.get(s, "gray") for s in saint_names]

        bars = ax_bar.barh(range(len(top_n)), scores[::-1], color=bar_colors[::-1], edgecolor="white")
        ax_bar.set_yticks(range(len(top_n)))
        ax_bar.set_yticklabels(labels[::-1], fontsize=7)
        ax_bar.set_xlabel("% of models where prediction is unstable", fontsize=9)
        ax_bar.set_title("Top 15 Most Unstable Images", fontsize=11)
        ax_bar.set_xlim(0, 105)
        ax_bar.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        for bar, score in zip(bars, scores[::-1]):
            ax_bar.text(score + 1, bar.get_y() + bar.get_height() / 2,
                        f"{score:.0f}%", va="center", fontsize=7)

        # Legend for saint colors
        patches = [mpatches.Patch(color=saint_color_map.get(s, "gray"), label=s) for s in saints]
        ax_bar.legend(handles=patches, fontsize=7, loc="lower right", title="Saint", title_fontsize=8)

    # ---- Pie chart: saint distribution of unstable images ----
    saint_unstable_count = defaultdict(int)
    for score, count, total, name, dataset, saint in scored:
        saint_unstable_count[saint] += count  # weighted by number of unstable models

    if saint_unstable_count:
        pie_labels = list(saint_unstable_count.keys())
        pie_values = list(saint_unstable_count.values())
        pie_colors = [saint_color_map.get(s, "gray") for s in pie_labels]
        wedges, texts, autotexts = ax_pie.pie(
            pie_values, labels=pie_labels, colors=pie_colors,
            autopct="%1.0f%%", startangle=140,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax_pie.set_title("Share of Instability by Saint\n(weighted by model count)", fontsize=11)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
