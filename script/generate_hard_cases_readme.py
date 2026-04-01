#!/usr/bin/env python3
"""
Generate hard_cases.md:
  - Top 5 most unstable images in a single combined table (images as rows, models as columns)
  - Class-level confusion tables
  - Generation analysis section (appended by analyze_generations.py)
"""

import json
from pathlib import Path
from collections import defaultdict

CONSISTENCY_DIR = Path(__file__).parent.parent / "dataset" / "consistency"
OUTPUT_PATH = CONSISTENCY_DIR / "hard_cases.md"

# LLM models only for the main table (CLIP/SigLIP summarised separately)
LLM_MODELS_ORDERED = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-11-20",
    "gpt-5-mini-2025-08-07",
    "gpt-5.2-2025-12-11",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

LLM_COL_HEADERS = {
    "gpt-4o-mini-2024-07-18":         "GPT-4o Mini",
    "gpt-4o-2024-11-20":              "GPT-4o",
    "gpt-5-mini-2025-08-07":          "GPT-5 Mini",
    "gpt-5.2-2025-12-11":             "GPT-5.2",
    "gemini-2.5-flash-lite":          "Gem 2.5 Flash Lite",
    "gemini-2.5-flash":               "Gem 2.5 Flash",
    "gemini-2.5-pro":                 "Gem 2.5 Pro",
    "gemini-3.1-flash-lite-preview":  "Gem 3.1 Flash Lite",
    "gemini-3-flash-preview":         "Gem 3 Flash",
    "gemini-3.1-pro-preview":         "Gem 3.1 Pro",
}

VISION_MODELS = [
    "clip-vit-base-patch16", "clip-vit-base-patch32", "clip-vit-large-patch14",
    "siglip-base-patch16-512", "siglip-large-patch16-384", "siglip-so400m-patch14-384",
]

ALL_MODELS = LLM_MODELS_ORDERED + VISION_MODELS

SAINT_LABELS = {
    "11H(JEROME)":           "Jerome",
    "11H(PETER)":            "Peter",
    "11H(JOHN THE BAPTIST)": "John the Baptist",
    "11H(FRANCIS)":          "Francis",
    "11HH(MARY MAGDALENE)":  "Mary Magdalene",
}

# Short abbreviations for predictions inside table cells
SAINT_ABBR = {
    "11H(JEROME)":              "JER",
    "11H(PETER)":               "PET",
    "11H(JOHN THE BAPTIST)":    "JTB",
    "11H(JOHN)":                "JOH",
    "11H(FRANCIS)":             "FRA",
    "11HH(MARY MAGDALENE)":     "MM",
    "11HH(CATHERINE)":          "CAT",
    "11F(MARY)":                "MARY",
    "11H(ANTONY ABBOT)":        "AA",
    "11H(JOSEPH)":              "JOS",
    "11H(SEBASTIAN)":           "SEB",
    "11H(PAUL)":                "PAU",
    "11H(LUKE)":                "LUK",
    "11H(DOMINIC)":             "DOM",
    "11H(ANTONY OF PADUA)":     "AoP",
}


def abbr(label):
    if label is None:
        return "—"
    return SAINT_ABBR.get(label, label[:6])


def label_short(label):
    if label is None:
        return "—"
    if label.startswith("11HH("):
        return label[5:-1]
    if label.startswith("11H("):
        return label[4:-1]
    return label


def load_model_predictions(model_name):
    model_dir = CONSISTENCY_DIR / model_name
    preds = defaultdict(dict)
    gts = {}
    for test in ["test_1", "test_2", "test_3"]:
        json_path = model_dir / test / "consistency_data.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            pairs = json.load(f)
        for pair in pairs:
            for img in pair:
                key = (img["name"], img["dataset"])
                preds[key][test] = img.get("predicted")
                if img.get("ground_truth") and key not in gts:
                    gts[key] = img["ground_truth"]
    return preds, gts


def load_pair_map():
    """Return dict: image_key -> partner_key, using first available model/test."""
    pair_map = {}
    for model in ALL_MODELS:
        jp = CONSISTENCY_DIR / model / "test_1" / "consistency_data.json"
        if not jp.exists():
            continue
        with open(jp) as f:
            pairs = json.load(f)
        for pair in pairs:
            if len(pair) == 2:
                k0 = (pair[0]["name"], pair[0]["dataset"])
                k1 = (pair[1]["name"], pair[1]["dataset"])
                # keep the first seen partner (pairs are stable across models)
                pair_map.setdefault(k0, k1)
                pair_map.setdefault(k1, k0)
        break  # one model is enough
    return pair_map


# ─── Load all data ────────────────────────────────────────────────────────────
all_preds = {}
all_gts = {}
image_total_models = defaultdict(int)
image_unstable_models = defaultdict(int)
image_model_preds = defaultdict(dict)  # (name,ds) -> {model -> {test -> pred}}

for model in ALL_MODELS:
    preds, gts = load_model_predictions(model)
    if not preds:
        continue
    all_preds[model] = preds
    all_gts.update(gts)
    for key, test_preds in preds.items():
        vals = [v for v in test_preds.values() if v is not None]
        if len(vals) < 2:
            continue
        image_total_models[key] += 1
        if len(set(vals)) > 1:
            image_unstable_models[key] += 1
        image_model_preds[key][model] = test_preds

# ─── Top 5 most unstable images ───────────────────────────────────────────────
scored = []
for key, n_unstable in image_unstable_models.items():
    n_total = image_total_models[key]
    if n_total >= 3:
        scored.append((n_unstable / n_total, n_unstable, n_total, key))
scored.sort(reverse=True)
top5_keys = [key for _, _, _, key in scored[:5]]

# ─── Pair relationships ───────────────────────────────────────────────────────
pair_map = load_pair_map()

# ─── Pair-level stats ────────────────────────────────────────────────────────
# For each (key_a, key_b) pair, across all models × tests count:
#   pair_total     : (model, test) slots where both images have a prediction
#   pair_consistent: both images predicted with the same class
#   pair_correct   : both images predicted with the same class == ground truth
pair_stats = {}   # (key_a, key_b) -> {total, consistent, correct}

def compute_pair_stats(key_a, key_b):
    gt_a = all_gts.get(key_a, "")
    gt_b = all_gts.get(key_b, "")
    total = consistent = correct = 0
    for model, preds in all_preds.items():
        tp_a = preds.get(key_a, {})
        tp_b = preds.get(key_b, {})
        for test in ["test_1", "test_2", "test_3"]:
            pa = tp_a.get(test)
            pb = tp_b.get(test)
            if pa is None or pb is None:
                continue
            total += 1
            if pa == pb:
                consistent += 1
                if pa == gt_a and pa == gt_b:
                    correct += 1
    return {"total": total, "consistent": consistent, "correct": correct}

# ─── Confusion data ───────────────────────────────────────────────────────────
confusion = defaultdict(lambda: defaultdict(int))
gt_total_preds = defaultdict(int)

for model, preds in all_preds.items():
    for key, test_preds in preds.items():
        gt = all_gts.get(key)
        if not gt:
            continue
        for pred in test_preds.values():
            if pred is None:
                continue
            gt_total_preds[gt] += 1
            if pred != gt:
                confusion[gt][pred] += 1

# ─── Build README ─────────────────────────────────────────────────────────────
lines = []
lines.append("# Hard Consistency Cases")
lines.append("")
lines.append(
    "This document identifies the images and saint classes that are most difficult "
    "to predict consistently across repeated model runs."
)
lines.append("")
lines.append(
    "> **Instability** = a model gives different predictions for the same image "
    "across `test_1`, `test_2`, `test_3`. The instability rate is the fraction "
    "of all models where this occurs for a given image."
)
lines.append("")

# ─── Section 1: Top 5 in a single table ──────────────────────────────────────
lines.append("## Top 5 Most Unstable Image Pairs")
lines.append("")
lines.append(
    "Each row shows a pair of images tested together. "
    "Both metrics are counted across all models and all 3 runs (one observation = one model × one test run)."
)
lines.append("")

def ratio(n, total):
    pct = n / total * 100 if total else 0
    return f"{n}/{total} ({pct:.0f}%)"


lines.append("| # | Image A | Image B | Ground Truth | Both predicted same class | Both predicted same class (correct) |")
lines.append("|---|---------|---------|-------------|--------------------------|-------------------------------------|")

for rank, key_a in enumerate(top5_keys, 1):
    key_b  = pair_map.get(key_a)
    name_a, ds_a = key_a
    gt_a   = all_gts.get(key_a, "")
    gt_short_a = SAINT_LABELS.get(gt_a, gt_a)
    img_a  = f"![{name_a}](example/{ds_a}_{name_a}.jpg)"

    if key_b:
        name_b, ds_b = key_b
        gt_b        = all_gts.get(key_b, "")
        gt_short_b  = SAINT_LABELS.get(gt_b, gt_b)
        img_b       = f"![{name_b}](example/{ds_b}_{name_b}.jpg)"
        gt_cell     = gt_short_a if gt_short_a == gt_short_b else f"{gt_short_a} / {gt_short_b}"
        ps          = compute_pair_stats(key_a, key_b)
        consistent  = ratio(ps["consistent"], ps["total"])
        correct     = ratio(ps["correct"],    ps["total"])
    else:
        img_b      = "—"
        gt_cell    = gt_short_a
        consistent = "—"
        correct    = "—"

    lines.append(f"| {rank} | {img_a} | {img_b} | {gt_cell} | {consistent} | {correct} |")

lines.append("")
lines.append(
    "> **Both predicted same class**: number of (model, run) observations where the model assigned "
    "the same label to both images (count/total, %).  "
    "**Both predicted same class (correct)**: same, but the shared label also matches the ground truth."
)
lines.append("")

# ─── Section 2: Class-level confusion ────────────────────────────────────────
lines.append("---")
lines.append("")
lines.append("## Most Unstable Classes — Incorrect Prediction Distribution")
lines.append("")
lines.append(
    "For each saint class, this table shows how often incorrect predictions are made "
    "and what alternative classes models tend to predict instead. "
    "Counts aggregate all models, all test runs, and all images of that class."
)
lines.append("")

saints_by_errors = sorted(confusion.keys(), key=lambda gt: sum(confusion[gt].values()), reverse=True)

lines.append("| Saint (Ground Truth) | Total Predictions | Wrong Predictions | Error Rate | Top Wrong Predictions |")
lines.append("|----------------------|-------------------|-------------------|------------|-----------------------|")

for gt in saints_by_errors:
    total = gt_total_preds[gt]
    wrong_map = confusion[gt]
    n_wrong = sum(wrong_map.values())
    error_rate = n_wrong / total * 100 if total else 0
    top_wrong = sorted(wrong_map.items(), key=lambda x: x[1], reverse=True)[:4]
    top_str = ", ".join(f"**{label_short(p)}** ({c})" for p, c in top_wrong)
    lines.append(f"| {gt} ({SAINT_LABELS.get(gt, gt)}) | {total} | {n_wrong} | {error_rate:.1f}% | {top_str} |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("### Detailed Confusion per Saint")
lines.append("")

for gt in saints_by_errors:
    wrong_map = confusion[gt]
    if not wrong_map:
        continue
    total = gt_total_preds[gt]
    n_wrong = sum(wrong_map.values())
    lines.append(f"#### {gt} ({SAINT_LABELS.get(gt, gt)})")
    lines.append("")
    lines.append(f"Total predictions: **{total}** — Wrong: **{n_wrong}** ({n_wrong/total*100:.1f}%)")
    lines.append("")
    lines.append("| Predicted As | Count | % of Errors |")
    lines.append("|--------------|-------|-------------|")
    for pred, count in sorted(wrong_map.items(), key=lambda x: x[1], reverse=True):
        pct = count / n_wrong * 100 if n_wrong else 0
        lines.append(f"| {pred} ({label_short(pred)}) | {count} | {pct:.1f}% |")
    lines.append("")

OUTPUT_PATH.write_text("\n".join(lines))
print(f"Written → {OUTPUT_PATH}")
