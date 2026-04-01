#!/usr/bin/env python3
"""
Investigate why GPT-4o/5.2 regress on ICONCLASS while GPT-4o Mini is weak,
and what well-performing Gemini models do differently.

Focus models:
  - gpt-4o-2024-11-20     (legacy, decent ~72%)
  - gpt-5.2-2025-12-11    (next-gen, regressed ~64%)
  - gpt-4o-mini-2024-07-18 (legacy, weak ~56%)
  - gemini-3-flash-preview (next-gen, strong ~93%)

Analyses:
  1. Per-class accuracy comparison — which classes collapsed?
  2. Confusion flow: where do GPT-5.2 new errors go?
  3. Image-level regression drill-down: what did GPT-4o get right that 5.2 lost?
  4. Confidence vs correctness: are wrong answers high-confidence?
  5. Class-level difficulty ranking: easy/hard classes per model
  6. Cross-model agreement matrix: do GPT models agree on errors?

Outputs: dataset/shift_analysis/
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

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent
DATASETS = ["ICONCLASS"]
TESTS = ["test_1", "test_2", "test_3"]

FOCUS_MODELS = {
    "gpt-4o-2024-11-20":      {"label": "GPT-4o (Nov)",   "color": "#E8927C", "gen": "Legacy"},
    "gpt-5.2-2025-12-11":     {"label": "GPT-5.2",        "color": "#C0392B", "gen": "Next-Gen"},
    "gpt-4o-mini-2024-07-18": {"label": "GPT-4o Mini",    "color": "#F5B041", "gen": "Legacy"},
    "gemini-3-flash-preview":  {"label": "Gem3 Flash",     "color": "#5DA271", "gen": "Next-Gen"},
}
MODEL_ORDER = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20",
               "gpt-5.2-2025-12-11", "gemini-3-flash-preview"]

def label(m):
    return FOCUS_MODELS[m]["label"]

def color(m):
    return FOCUS_MODELS[m]["color"]


# ─── Data loading ────────────────────────────────────────────────────────────

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


def load_predictions(model, dataset, test, classes):
    model_dir = BASE_DIR / test / dataset / model
    probs_path = model_dir / "probs.npy"
    ids_path = model_dir / "image_ids.txt"
    if not probs_path.exists():
        return {}

    probs = np.load(probs_path)
    if ids_path.exists():
        images = ids_path.read_text().splitlines()
    else:
        test_path = BASE_DIR / "dataset" / f"{dataset}-data" / "2_test.txt"
        images = test_path.read_text().splitlines()

    class_ids = [c[0] for c in classes]
    result = {}
    for i, img in enumerate(images[:len(probs)]):
        p = probs[i].astype(float)
        pred_idx = int(p.argmax())
        pred_cls = class_ids[pred_idx] if pred_idx < len(class_ids) else None
        sorted_p = np.sort(p)[::-1]
        result[img] = {
            "pred": pred_cls,
            "probs": p,
            "confidence": float(sorted_p[0]),
            "margin": float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0]),
            "entropy": float(-np.sum(np.clip(p, 1e-12, 1) * np.log2(np.clip(p, 1e-12, 1)))),
        }
    return result


# ─── Analysis 1: Per-class accuracy comparison ──────────────────────────────

def analyze_perclass_accuracy(all_preds, gt, classes, dataset, test):
    out = OUTPUT_DIR / "1_perclass_accuracy" / dataset / test
    out.mkdir(parents=True, exist_ok=True)
    class_ids = [c[0] for c in classes]
    class_labels = {c[0]: c[1] for c in classes}

    records = []
    for model in MODEL_ORDER:
        preds = all_preds.get(model, {})
        for cid in class_ids:
            imgs_in_class = [img for img, g in gt.items() if g == cid and img in preds]
            if not imgs_in_class:
                continue
            correct = sum(1 for img in imgs_in_class if preds[img]["pred"] == cid)
            records.append({
                "model": label(model),
                "model_id": model,
                "class_id": cid,
                "class_label": class_labels[cid],
                "n_images": len(imgs_in_class),
                "correct": correct,
                "recall": correct / len(imgs_in_class) * 100,
            })

    df = pd.DataFrame(records)
    df.to_csv(out / "perclass_recall.csv", index=False)

    # Pivot: class × model recall
    pivot = df.pivot_table(index="class_id", columns="model", values="recall")
    pivot = pivot.reindex(columns=[label(m) for m in MODEL_ORDER])

    # Sort by GPT-5.2 regression (GPT-4o recall - GPT-5.2 recall)
    if label("gpt-4o-2024-11-20") in pivot.columns and label("gpt-5.2-2025-12-11") in pivot.columns:
        pivot["regression"] = pivot[label("gpt-4o-2024-11-20")] - pivot[label("gpt-5.2-2025-12-11")]
        pivot = pivot.sort_values("regression", ascending=False)
        pivot = pivot.drop(columns=["regression"])

    short_labels = [class_labels.get(c, c).split(",")[0][:25] for c in pivot.index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                yticklabels=short_labels, ax=ax,
                cbar_kws={"label": "Recall (%)"}, linewidths=0.5, linecolor="#eee")
    ax.set_title(f"Per-Class Recall (%) — {dataset} / {test}\n"
                 f"Sorted by GPT-4o → GPT-5.2 regression (top = biggest drop)",
                 fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out / "perclass_recall_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    w = 0.2
    for i, model in enumerate(MODEL_ORDER):
        vals = pivot[label(model)].values
        bars = ax.bar(x + i * w, vals, w, label=label(model), color=color(model),
                      edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_title(f"Per-Class Recall Comparison — {dataset} / {test}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "perclass_recall_bars.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  [1] Per-class accuracy → {out}")
    return df


# ─── Analysis 2: Confusion flow (GPT-4o → GPT-5.2 regressions) ─────────────

def analyze_confusion_flow(all_preds, gt, classes, dataset, test):
    out = OUTPUT_DIR / "2_confusion_flow" / dataset / test
    out.mkdir(parents=True, exist_ok=True)
    class_ids = [c[0] for c in classes]
    class_labels = {c[0]: c[1].split(",")[0][:20] for c in classes}

    pairs = [
        ("gpt-4o-2024-11-20", "gpt-5.2-2025-12-11", "GPT-4o → GPT-5.2"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "GPT-4o Mini → GPT-4o"),
    ]

    for old_m, new_m, pair_label in pairs:
        old_p = all_preds.get(old_m, {})
        new_p = all_preds.get(new_m, {})
        common = [img for img in set(old_p) & set(new_p) if img in gt]
        if not common:
            continue

        # Transition matrix
        cm = np.zeros((len(class_ids), len(class_ids)), dtype=int)
        regressions = []
        improvements = []
        for img in common:
            g = gt[img]
            op = old_p[img]["pred"]
            np_ = new_p[img]["pred"]
            if op in class_ids and np_ in class_ids:
                cm[class_ids.index(op), class_ids.index(np_)] += 1
            old_ok = op == g
            new_ok = np_ == g
            if old_ok and not new_ok:
                regressions.append({"image": img, "true_class": g,
                                    "old_pred": op, "new_pred": np_,
                                    "new_confidence": new_p[img]["confidence"]})
            elif not old_ok and new_ok:
                improvements.append({"image": img, "true_class": g,
                                     "old_pred": op, "new_pred": np_})

        safe = f"{old_m}_to_{new_m}"

        # Save regression cases
        if regressions:
            reg_df = pd.DataFrame(regressions)
            reg_df.to_csv(out / f"regressions_{safe}.csv", index=False)

            # Where do regressions go? (true class → wrong new prediction)
            reg_flow = reg_df.groupby(["true_class", "new_pred"]).size().reset_index(name="count")
            reg_flow = reg_flow.sort_values("count", ascending=False)
            reg_flow["true_label"] = reg_flow["true_class"].map(class_labels)
            reg_flow["new_label"] = reg_flow["new_pred"].map(class_labels)
            reg_flow.to_csv(out / f"regression_flow_{safe}.csv", index=False)

            # Sankey-style: top regression flows as horizontal bar
            top_flows = reg_flow.head(15)
            top_flows = top_flows.copy()
            top_flows["flow"] = top_flows["true_label"] + " → " + top_flows["new_label"]

            fig, ax = plt.subplots(figsize=(10, max(5, len(top_flows) * 0.4)))
            bars = ax.barh(range(len(top_flows)), top_flows["count"].values,
                           color="#E74C3C", edgecolor="white", height=0.7)
            ax.set_yticks(range(len(top_flows)))
            ax.set_yticklabels(top_flows["flow"].values, fontsize=9)
            for bar, v in zip(bars, top_flows["count"].values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=9, fontweight="bold")
            ax.set_xlabel("# of images", fontsize=11)
            ax.set_title(f"Top Regression Flows: {pair_label}\n"
                         f"{dataset} / {test} — True class → New wrong prediction",
                         fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out / f"regression_flows_{safe}.png", dpi=200, bbox_inches="tight")
            plt.close()

        # Transition heatmap (row-normalised)
        short_ids = [class_labels.get(c, c) for c in class_ids]
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=short_ids, yticklabels=short_ids,
                    ax=ax, linewidths=0.3, cbar_kws={"label": "% of old row"})
        ax.set_xlabel(f"New model ({label(new_m)}) prediction", fontsize=11)
        ax.set_ylabel(f"Old model ({label(old_m)}) prediction", fontsize=11)
        ax.set_title(f"Prediction Transition: {pair_label}\n{dataset} / {test}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out / f"transition_heatmap_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Summary
        n_reg = len(regressions)
        n_imp = len(improvements)
        n_total = len(common)
        print(f"  [2] {pair_label}: {n_reg} regressions, {n_imp} improvements "
              f"(net={n_imp - n_reg:+d}) out of {n_total} images")

    print(f"  [2] Confusion flow → {out}")


# ─── Analysis 3: Confidence analysis — are wrong answers overconfident? ──────

def analyze_confidence_patterns(all_preds, gt, classes, dataset, test):
    out = OUTPUT_DIR / "3_confidence_patterns" / dataset / test
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for model in MODEL_ORDER:
        preds = all_preds.get(model, {})
        for img, pred_info in preds.items():
            if img not in gt:
                continue
            correct = int(pred_info["pred"] == gt[img])
            records.append({
                "model": label(model),
                "model_id": model,
                "image": img,
                "correct": correct,
                "confidence": pred_info["confidence"],
                "margin": pred_info["margin"],
                "entropy": pred_info["entropy"],
                "true_class": gt[img],
                "pred_class": pred_info["pred"],
            })

    df = pd.DataFrame(records)
    df.to_csv(out / "confidence_data.csv", index=False)

    # ── Confidence distribution: correct vs wrong per model ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Confidence When Right vs Wrong — {dataset} / {test}",
                 fontsize=14, fontweight="bold")

    for ax, model in zip(axes.flat, MODEL_ORDER):
        sub = df[df["model_id"] == model]
        for correct_val, c, lbl in [(1, "#4CAF50", "Correct"), (0, "#E74C3C", "Wrong")]:
            vals = sub[sub["correct"] == correct_val]["confidence"]
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.5, color=c, label=f"{lbl} (n={len(vals)})",
                        density=True, edgecolor=c, linewidth=0.8)
        ax.set_title(label(model), fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence (max prob)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "confidence_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ── Overconfident wrong answers: confidence > 0.9 but wrong ──
    overconf = df[(df["correct"] == 0) & (df["confidence"] > 0.9)]
    overconf_summary = (overconf.groupby(["model", "model_id"])
                                .size().reset_index(name="overconfident_wrong"))
    total_wrong = (df[df["correct"] == 0].groupby(["model", "model_id"])
                                         .size().reset_index(name="total_wrong"))
    overconf_summary = overconf_summary.merge(total_wrong)
    overconf_summary["pct_overconfident"] = (overconf_summary["overconfident_wrong"] /
                                              overconf_summary["total_wrong"] * 100)
    overconf_summary.to_csv(out / "overconfident_wrong.csv", index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    oc_ordered = overconf_summary.set_index("model_id").reindex(MODEL_ORDER).reset_index()
    colors_list = [color(m) for m in oc_ordered["model_id"]]
    bars = ax.bar(oc_ordered["model"], oc_ordered["pct_overconfident"],
                  color=colors_list, width=0.6, edgecolor="white")
    for bar, v, n in zip(bars, oc_ordered["pct_overconfident"], oc_ordered["overconfident_wrong"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%\n({n} imgs)", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("% of wrong answers with conf > 0.9", fontsize=11)
    ax.set_title(f"Overconfident Wrong Predictions — {dataset} / {test}\n"
                 f"(confidence > 0.9 but prediction is wrong)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "overconfident_wrong_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  [3] Confidence patterns → {out}")
    return df


# ─── Analysis 4: Class difficulty ranking ────────────────────────────────────

def analyze_class_difficulty(all_preds, gt, classes, dataset, test):
    out = OUTPUT_DIR / "4_class_difficulty" / dataset / test
    out.mkdir(parents=True, exist_ok=True)
    class_ids = [c[0] for c in classes]
    class_labels = {c[0]: c[1].split(",")[0][:25] for c in classes}

    records = []
    for cid in class_ids:
        imgs = [img for img, g in gt.items() if g == cid]
        row = {"class_id": cid, "class_label": class_labels.get(cid, cid), "n_images": len(imgs)}
        for model in MODEL_ORDER:
            preds = all_preds.get(model, {})
            correct = sum(1 for img in imgs if img in preds and preds[img]["pred"] == cid)
            row[f"{label(model)}_recall"] = correct / len(imgs) * 100 if imgs else 0
            # Average confidence for this class
            confs = [preds[img]["confidence"] for img in imgs if img in preds]
            row[f"{label(model)}_avg_conf"] = np.mean(confs) if confs else 0
        records.append(row)

    df = pd.DataFrame(records)

    # Compute "GPT difficulty" = avg recall across GPT models
    gpt_cols = [c for c in df.columns if c.endswith("_recall") and "GPT" in c]
    gem_cols = [c for c in df.columns if c.endswith("_recall") and "Gem" in c]
    df["gpt_avg_recall"] = df[gpt_cols].mean(axis=1)
    df["gem_avg_recall"] = df[gem_cols].mean(axis=1)
    df["gpt_gem_gap"] = df["gem_avg_recall"] - df["gpt_avg_recall"]
    df = df.sort_values("gpt_gem_gap", ascending=False)
    df.to_csv(out / "class_difficulty.csv", index=False)

    # Scatter: GPT avg recall vs Gemini recall
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(df["gpt_avg_recall"], df["gem_avg_recall"],
               s=df["n_images"] * 2, alpha=0.7, c="#3498DB", edgecolors="white", linewidth=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["class_label"], (row["gpt_avg_recall"], row["gem_avg_recall"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9, alpha=0.8)
    # Diagonal
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("GPT avg recall (%)", fontsize=12)
    ax.set_ylabel("Gemini 3 Flash recall (%)", fontsize=12)
    ax.set_title(f"Class Difficulty: GPT vs Gemini — {dataset} / {test}\n"
                 f"(bubble size = # images; above diagonal = Gemini better)", fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "gpt_vs_gemini_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Stacked: gap per class
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    ax.bar(x, df["gpt_avg_recall"].values, color="#E8927C", label="GPT avg recall", width=0.7)
    ax.bar(x, df["gpt_gem_gap"].values, bottom=df["gpt_avg_recall"].values,
           color="#5DA271", label="Gemini advantage", width=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_label"].values, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_title(f"Gemini Advantage Over GPT Models by Class — {dataset} / {test}\n"
                 f"(sorted by gap size)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "gemini_advantage_by_class.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  [4] Class difficulty → {out}")


# ─── Analysis 5: Cross-model agreement on errors ────────────────────────────

def analyze_error_agreement(all_preds, gt, classes, dataset, test):
    out = OUTPUT_DIR / "5_error_agreement" / dataset / test
    out.mkdir(parents=True, exist_ok=True)

    # For each image: which models got it wrong?
    common_imgs = set.intersection(*(set(all_preds[m].keys()) for m in MODEL_ORDER if m in all_preds))
    common_imgs = [img for img in common_imgs if img in gt]

    error_matrix = []
    for img in common_imgs:
        row = {"image": img, "true_class": gt[img]}
        for model in MODEL_ORDER:
            pred = all_preds[model][img]["pred"]
            row[f"{label(model)}_pred"] = pred
            row[f"{label(model)}_correct"] = int(pred == gt[img])
        error_matrix.append(row)

    edf = pd.DataFrame(error_matrix)

    # Pairwise error overlap (Jaccard of error sets)
    correct_cols = [f"{label(m)}_correct" for m in MODEL_ORDER]
    model_labels = [label(m) for m in MODEL_ORDER]
    n = len(model_labels)
    jaccard = np.zeros((n, n))
    overlap_n = np.zeros((n, n), dtype=int)

    for i in range(n):
        errors_i = set(edf[edf[correct_cols[i]] == 0]["image"])
        for j in range(n):
            errors_j = set(edf[edf[correct_cols[j]] == 0]["image"])
            union = errors_i | errors_j
            inter = errors_i & errors_j
            jaccard[i, j] = len(inter) / len(union) * 100 if union else 0
            overlap_n[i, j] = len(inter)

    # Jaccard heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(jaccard, annot=True, fmt=".0f", cmap="YlOrRd",
                xticklabels=model_labels, yticklabels=model_labels,
                ax=ax, linewidths=0.5, cbar_kws={"label": "Jaccard similarity (%) of error sets"})
    ax.set_title(f"Error Set Overlap — {dataset} / {test}\n"
                 f"(high = models make similar mistakes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out / "error_jaccard_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Count images wrong by N models
    edf["n_models_wrong"] = sum(1 - edf[c] for c in correct_cols)

    n_wrong_counts = edf["n_models_wrong"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(n_wrong_counts.index.astype(int), n_wrong_counts.values,
                  color="#3498DB", edgecolor="white", width=0.7)
    for bar, v in zip(bars, n_wrong_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("# of models that got image wrong", fontsize=11)
    ax.set_ylabel("# of images", fontsize=11)
    ax.set_title(f"Image Difficulty Distribution — {dataset} / {test}\n"
                 f"(0 = all correct, 4 = universally hard)", fontsize=12)
    ax.set_xticks(range(5))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "images_by_difficulty.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Images only Gemini gets right (wrong for all 3 GPT models)
    gpt_correct_cols = [f"{label(m)}_correct" for m in MODEL_ORDER if "gpt" in m]
    gem_col = f"{label('gemini-3-flash-preview')}_correct"
    gemini_only = edf[(edf[gpt_correct_cols].sum(axis=1) == 0) & (edf[gem_col] == 1)]
    gemini_only_classes = gemini_only["true_class"].value_counts()

    if not gemini_only_classes.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        class_labels_map = {c[0]: c[1].split(",")[0][:25] for c in classes}
        short = [class_labels_map.get(c, c) for c in gemini_only_classes.index]
        ax.barh(range(len(gemini_only_classes)), gemini_only_classes.values,
                color="#5DA271", edgecolor="white", height=0.7)
        ax.set_yticks(range(len(gemini_only_classes)))
        ax.set_yticklabels(short, fontsize=9)
        for i, v in enumerate(gemini_only_classes.values):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("# of images", fontsize=11)
        ax.set_title(f"Images Only Gemini 3 Flash Gets Right (all GPT models wrong)\n"
                     f"{dataset} / {test} — {len(gemini_only)} images total", fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "gemini_only_correct.png", dpi=200, bbox_inches="tight")
        plt.close()

    edf.to_csv(out / "error_matrix.csv", index=False)
    print(f"  [5] Error agreement → {out}")


# ─── Analysis 6: GPT-4o → GPT-5.2 regression deep dive ──────────────────────

def analyze_regression_deepdive(all_preds, gt, classes, dataset, test):
    """Per-image: what GPT-4o got right that GPT-5.2 lost, with confidence context."""
    out = OUTPUT_DIR / "6_regression_deepdive" / dataset / test
    out.mkdir(parents=True, exist_ok=True)
    class_labels = {c[0]: c[1].split(",")[0][:25] for c in classes}

    old_m = "gpt-4o-2024-11-20"
    new_m = "gpt-5.2-2025-12-11"
    gem_m = "gemini-3-flash-preview"
    old_p = all_preds.get(old_m, {})
    new_p = all_preds.get(new_m, {})
    gem_p = all_preds.get(gem_m, {})
    common = [img for img in set(old_p) & set(new_p) if img in gt]

    records = []
    for img in common:
        g = gt[img]
        old_ok = old_p[img]["pred"] == g
        new_ok = new_p[img]["pred"] == g
        gem_ok = gem_p[img]["pred"] == g if img in gem_p else None

        if old_ok and not new_ok:
            category = "regression"
        elif not old_ok and new_ok:
            category = "improvement"
        elif old_ok and new_ok:
            category = "stable_correct"
        else:
            category = "stable_wrong"

        records.append({
            "image": img,
            "true_class": g,
            "true_label": class_labels.get(g, g),
            "category": category,
            "gpt4o_pred": old_p[img]["pred"],
            "gpt4o_conf": old_p[img]["confidence"],
            "gpt52_pred": new_p[img]["pred"],
            "gpt52_conf": new_p[img]["confidence"],
            "gpt52_new_pred_label": class_labels.get(new_p[img]["pred"], new_p[img]["pred"]),
            "gemini_correct": gem_ok,
            "gemini_pred": gem_p[img]["pred"] if img in gem_p else None,
        })

    rdf = pd.DataFrame(records)
    rdf.to_csv(out / "regression_deepdive.csv", index=False)

    # Category counts
    cat_counts = rdf["category"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"GPT-4o → GPT-5.2 Transition Analysis — {dataset} / {test}",
                 fontsize=14, fontweight="bold")

    # Left: pie/bar of categories
    ax = axes[0]
    cat_order = ["stable_correct", "stable_wrong", "improvement", "regression"]
    cat_colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
    cat_labels_nice = ["Stable correct", "Stable wrong", "Improvement", "Regression"]
    vals = [cat_counts.get(c, 0) for c in cat_order]
    bars = ax.bar(cat_labels_nice, vals, color=cat_colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        pct = v / len(rdf) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("# of images", fontsize=11)
    ax.set_title("Image-Level Transition Categories", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Right: for regressions, does Gemini get them right?
    ax = axes[1]
    reg_only = rdf[rdf["category"] == "regression"]
    if not reg_only.empty and "gemini_correct" in reg_only.columns:
        gem_correct_on_reg = reg_only["gemini_correct"].sum()
        gem_wrong_on_reg = len(reg_only) - gem_correct_on_reg
        ax.bar(["Gemini correct", "Gemini also wrong"],
               [gem_correct_on_reg, gem_wrong_on_reg],
               color=["#5DA271", "#E74C3C"], edgecolor="white", width=0.5)
        for i, v in enumerate([gem_correct_on_reg, gem_wrong_on_reg]):
            ax.text(i, v + 1, str(int(v)), ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel("# of GPT-5.2 regressions", fontsize=11)
        ax.set_title(f"Do GPT-5.2 regressions also fool Gemini?\n"
                     f"({int(gem_correct_on_reg)}/{len(reg_only)} = Gemini still gets them right)",
                     fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "regression_deepdive.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Regression confidence analysis
    if not reg_only.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(reg_only["gpt52_conf"], bins=20, alpha=0.6, color="#FF9800",
                label=f"GPT-5.2 confidence on regressions (n={len(reg_only)})",
                edgecolor="#FF9800", density=True)
        ax.hist(reg_only["gpt4o_conf"], bins=20, alpha=0.6, color="#3498DB",
                label=f"GPT-4o confidence on same images (was correct)",
                edgecolor="#3498DB", density=True)
        ax.set_xlabel("Confidence (max prob)", fontsize=11)
        ax.set_title(f"Confidence on Regressed Images — {dataset} / {test}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "regression_confidence.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"  [6] Regression deep dive → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        data_dir = BASE_DIR / "dataset" / f"{dataset}-data"
        if not data_dir.exists():
            print(f"[SKIP] {dataset}: data dir not found")
            continue

        gt = load_ground_truth(dataset)

        for test in TESTS:
            print(f"\n{'='*60}")
            print(f"  {dataset} / {test}")
            print(f"{'='*60}")

            classes = load_classes(dataset, test)
            all_preds = {}
            for model in MODEL_ORDER:
                preds = load_predictions(model, dataset, test, classes)
                if preds:
                    all_preds[model] = preds
                    n_correct = sum(1 for img, p in preds.items()
                                    if img in gt and p["pred"] == gt[img])
                    n_total = sum(1 for img in preds if img in gt)
                    print(f"    {label(model):15s}: {n_correct}/{n_total} = "
                          f"{n_correct/n_total*100:.1f}%")
                else:
                    print(f"    {label(model):15s}: [NOT FOUND]")

            if len(all_preds) < 2:
                continue

            analyze_perclass_accuracy(all_preds, gt, classes, dataset, test)
            analyze_confusion_flow(all_preds, gt, classes, dataset, test)
            analyze_confidence_patterns(all_preds, gt, classes, dataset, test)
            analyze_class_difficulty(all_preds, gt, classes, dataset, test)
            analyze_error_agreement(all_preds, gt, classes, dataset, test)
            analyze_regression_deepdive(all_preds, gt, classes, dataset, test)

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
