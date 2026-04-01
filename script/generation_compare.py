#!/usr/bin/env python3
"""
Cross-generation accuracy comparison charts (print-friendly).

Produces three figures:
  1. generation_compare.png/.pdf   — slope chart (old vs new generation)
  2. generation_extremes.png/.pdf  — delta bar chart (accuracy change)
  3. generation_timeline.png/.pdf  — timeline (X = release month, Y = accuracy)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.patches import Patch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

TESTS = {"test_1": "Test 1", "test_2": "Test 2", "test_3": "Test 3"}
DATASETS = ["ArtDL", "ICONCLASS", "Wikidata"]

# Each model: folder name, display label, release date (for timeline)
MODELS = {
    "gpt-4o-mini-2024-07-18":       ("GPT-4o mini",           datetime(2024, 7, 18)),
    "gpt-4o-2024-11-20":            ("GPT-4o",                datetime(2024, 11, 20)),
    "gemini-2.5-flash-lite":        ("Gemini 2.5 Flash Lite", datetime(2025, 3, 25)),
    "gemini-2.5-flash":             ("Gemini 2.5 Flash",      datetime(2025, 3, 25)),
    "gemini-2.5-pro":               ("Gemini 2.5 Pro",        datetime(2025, 3, 25)),
    "gpt-5-mini-2025-08-07":        ("GPT-5 mini",            datetime(2025, 8, 7)),
    "gpt-5.2-2025-12-11":           ("GPT-5.2",               datetime(2025, 12, 11)),
    "gemini-3-flash-preview":       ("Gemini 3 Flash",        datetime(2026, 1, 21)),
    "gemini-3.1-flash-lite-preview": ("Gemini 3.1 Flash Lite", datetime(2026, 2, 5)),
    "gemini-3.1-pro-preview":       ("Gemini 3.1 Pro",        datetime(2026, 2, 5)),
}

# Families: old -> new
MODEL_FAMILIES = {
    "Gemini Flash Lite": ("gemini-2.5-flash-lite",  "gemini-3.1-flash-lite-preview"),
    "Gemini Flash":      ("gemini-2.5-flash",        "gemini-3-flash-preview"),
    "Gemini Pro":        ("gemini-2.5-pro",           "gemini-3.1-pro-preview"),
    "GPT Mini":          ("gpt-4o-mini-2024-07-18",   "gpt-5-mini-2025-08-07"),
    "GPT Full":          ("gpt-4o-2024-11-20",         "gpt-5.2-2025-12-11"),
}

FAMILY_SHORT = {
    "Gemini Flash Lite": "Gem. Flash Lite",
    "Gemini Flash":      "Gem. Flash",
    "Gemini Pro":        "Gem. Pro",
    "GPT Mini":          "GPT Mini",
    "GPT Full":          "GPT Full",
}

FAMILY_COLORS = {
    "Gemini Flash Lite": "#76B7B2",
    "Gemini Flash":      "#4E79A7",
    "Gemini Pro":        "#364B9A",
    "GPT Mini":          "#E15759",
    "GPT Full":          "#B07AA1",
}

DS_COLORS = {"ArtDL": "#4E79A7", "ICONCLASS": "#E15759", "Wikidata": "#59A14F"}

TEST_COLORS = {"test_1": "#4e79a7", "test_2": "#e15759", "test_3": "#59a14f"}


# ── Shared matplotlib style ─────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    })


DS_FOLDER = {"ArtDL": "ArtDL", "ICONCLASS": "ICONCLASS", "Wikidata": "wikidata"}


def load_accuracy(test_key, dataset, model_name):
    folder = DS_FOLDER.get(dataset, dataset)
    csv_path = BASE_DIR / test_key / folder / model_name / "summary_metrics.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    rows = df.loc[df["Model"] == model_name, "Accuracy"]
    if rows.empty:
        return None
    val = rows.values[0]
    if isinstance(val, str):
        return float(val.rstrip("%")) / 100.0
    return float(val)


# ── Chart 1: Slope chart ────────────────────────────────────────────────────

HIGHLIGHT_COLORS = {"GPT Mini": "#C0392B", "GPT Full": "#2471A3"}
HIGHLIGHT_PER_PANEL = {
    "Wikidata":  {"GPT Mini"},
    "ICONCLASS": {"GPT Full"},
}
MUTED = {
    "Gemini Flash Lite": "#C8C8C8",
    "Gemini Flash":      "#A0A0A0",
    "Gemini Pro":        "#787878",
    "GPT Mini":          "#B0B0B0",
    "GPT Full":          "#B0B0B0",
}


def _decollide_labels(labels, min_gap=0.018):
    """Push label y-positions apart so they don't overlap.
    Each label is a dict with 'y' (original position) and other data.
    Modifies 'y_display' in-place."""
    labels.sort(key=lambda l: l["y"])
    for i in range(1, len(labels)):
        if labels[i]["y_display"] - labels[i - 1]["y_display"] < min_gap:
            labels[i]["y_display"] = labels[i - 1]["y_display"] + min_gap


def plot_slopes():
    slope_data = {}
    for dataset in DATASETS:
        slope_data[dataset] = []
        for fam_name, (old_id, new_id) in MODEL_FAMILIES.items():
            old_accs, new_accs = [], []
            for t in TESTS:
                o = load_accuracy(t, dataset, old_id)
                n = load_accuracy(t, dataset, new_id)
                if o is not None:
                    old_accs.append(o)
                if n is not None:
                    new_accs.append(n)
            if old_accs and new_accs:
                slope_data[dataset].append({
                    "family": fam_name,
                    "old_mean": np.mean(old_accs), "new_mean": np.mean(new_accs),
                    "old_min": np.min(old_accs),   "old_max": np.max(old_accs),
                    "new_min": np.min(new_accs),   "new_max": np.max(new_accs),
                })

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    fig.suptitle("Cross-generation accuracy shift",
                 fontsize=13, fontweight="bold", y=0.99)

    for ax, dataset in zip(axes, DATASETS):
        entries = slope_data[dataset]
        hl_set = HIGHLIGHT_PER_PANEL.get(dataset, set())

        # Draw muted families first, then highlighted on top
        for highlighted in (False, True):
            for e in entries:
                is_hl = e["family"] in hl_set
                if is_hl != highlighted:
                    continue
                c = (HIGHLIGHT_COLORS[e["family"]] if is_hl
                     else MUTED.get(e["family"], "#B0B0B0"))
                lw = 2.2 if is_hl else 1.4
                alpha_bar = 0.5 if is_hl else 0.2
                zorder_line = 4 if is_hl else 2
                zorder_dot = 5 if is_hl else 3

                ax.plot([0, 1], [e["old_mean"], e["new_mean"]],
                        color=c, lw=lw, solid_capstyle="round", zorder=zorder_line)
                ax.plot([0, 0], [e["old_min"], e["old_max"]],
                        color=c, lw=1, alpha=alpha_bar, zorder=zorder_line - 1)
                ax.plot([1, 1], [e["new_min"], e["new_max"]],
                        color=c, lw=1, alpha=alpha_bar, zorder=zorder_line - 1)
                for x, val in [(0, e["old_mean"]), (1, e["new_mean"])]:
                    ax.scatter(x, val, color=c, s=36, zorder=zorder_dot,
                               edgecolors="white", linewidth=0.4)

        # Build labels with de-collision
        labels = []
        for e in entries:
            is_hl = e["family"] in hl_set
            c = (HIGHLIGHT_COLORS[e["family"]] if is_hl
                 else MUTED.get(e["family"], "#B0B0B0"))
            labels.append({
                "y": e["new_mean"], "y_display": e["new_mean"],
                "text": f"{FAMILY_SHORT[e['family']]}  {e['new_mean']:.0%}",
                "color": c,
                "bold": is_hl,
            })
        _decollide_labels(labels)

        for lbl in labels:
            ax.annotate(
                lbl["text"],
                xy=(1, lbl["y"]), xytext=(1.04, lbl["y_display"]),
                fontsize=7.5, va="center", color=lbl["color"],
                fontweight="bold" if lbl["bold"] else "normal",
                arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5)
                    if abs(lbl["y_display"] - lbl["y"]) > 0.005 else None,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Previous\ngeneration", "Current\ngeneration"], fontsize=9)
        ax.set_xlim(-0.15, 1.55)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.45, 1.02)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.grid(axis="y")

    axes[0].set_ylabel("Accuracy (avg over 3 tests, bars = range)", fontsize=9.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        plt.savefig(BASE_DIR / f"generation_compare.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: generation_compare.png / .pdf")


# ── Chart 2: Delta bar chart ────────────────────────────────────────────────

def plot_deltas():
    records = []
    for fam_name, (old_id, new_id) in MODEL_FAMILIES.items():
        for dataset in DATASETS:
            old_accs, new_accs = [], []
            for t in TESTS:
                o = load_accuracy(t, dataset, old_id)
                n = load_accuracy(t, dataset, new_id)
                if o is not None:
                    old_accs.append(o)
                if n is not None:
                    new_accs.append(n)
            if old_accs and new_accs:
                records.append({
                    "family": fam_name,
                    "dataset": dataset,
                    "delta": np.mean(new_accs) - np.mean(old_accs),
                })
    df = pd.DataFrame(records)

    families = list(MODEL_FAMILIES.keys())
    n_ds = len(DATASETS)
    bar_h = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    y_positions, y_labels = [], []

    for i, fam in enumerate(reversed(families)):
        base_y = i * (n_ds + 1) * bar_h
        for j, ds in enumerate(DATASETS):
            row = df[(df["family"] == fam) & (df["dataset"] == ds)]
            if row.empty:
                continue
            delta = row["delta"].values[0]
            y = base_y + j * bar_h
            ax.barh(y, delta * 100, height=bar_h * 0.85,
                    color=DS_COLORS[ds], alpha=0.8,
                    edgecolor="white", linewidth=0.5, zorder=3)
            sign = "+" if delta >= 0 else ""
            nudge = 0.3 if delta >= 0 else -0.3
            ha = "left" if delta >= 0 else "right"
            ax.text(delta * 100 + nudge, y, f"{sign}{delta*100:.1f}pp",
                    va="center", ha=ha, fontsize=8, fontweight="bold", color="#333")
        y_positions.append(base_y + (n_ds - 1) * bar_h / 2)
        y_labels.append(FAMILY_SHORT[fam])

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
    ax.axvline(0, color="#333", lw=0.8, zorder=2)
    ax.set_xlabel("Accuracy change (percentage points)", fontsize=10)
    ax.set_title("Generational Accuracy Change\n"
                 "(current - previous generation, averaged across 3 tests)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x")
    ax.legend(handles=[Patch(facecolor=DS_COLORS[ds], label=ds, alpha=0.8)
                       for ds in DATASETS],
              fontsize=9, frameon=False, loc="lower right",
              title="Dataset", title_fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(BASE_DIR / f"generation_extremes.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: generation_extremes.png / .pdf")


# ── Chart 3: Timeline ───────────────────────────────────────────────────────

def plot_timeline():
    """One row per test. X = release month, Y = accuracy.
    Each model is a point; models in the same family are connected."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Model Accuracy over Release Date",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, (test_key, test_label) in zip(axes, TESTS.items()):
        # Plot family connection lines first (behind dots)
        for fam_name, (old_id, new_id) in MODEL_FAMILIES.items():
            c = FAMILY_COLORS[fam_name]
            for dataset in DATASETS:
                old_acc = load_accuracy(test_key, dataset, old_id)
                new_acc = load_accuracy(test_key, dataset, new_id)
                if old_acc is not None and new_acc is not None:
                    old_date = MODELS[old_id][1]
                    new_date = MODELS[new_id][1]
                    ax.plot([old_date, new_date], [old_acc, new_acc],
                            color=c, lw=1, alpha=0.25, zorder=1)

        # Plot individual model dots
        for model_id, (label, date) in MODELS.items():
            for dataset in DATASETS:
                acc = load_accuracy(test_key, dataset, model_id)
                if acc is None:
                    continue
                # Determine family color
                fam_color = "#888"
                for fam_name, (old_id, new_id) in MODEL_FAMILIES.items():
                    if model_id in (old_id, new_id):
                        fam_color = FAMILY_COLORS[fam_name]
                        break
                marker = {"ArtDL": "o", "ICONCLASS": "s", "Wikidata": "D"}[dataset]
                ax.scatter(date, acc, color=fam_color, marker=marker,
                           s=38, zorder=3, edgecolors="white", linewidth=0.4)

        ax.set_ylabel("Accuracy", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0.40, 1.02)
        ax.set_title(test_label, fontsize=11, fontweight="bold", pad=6)
        ax.grid(axis="both")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")

    # Combined legend: families (colors) + datasets (markers)
    fam_handles = [plt.Line2D([0], [0], marker="o", color=FAMILY_COLORS[f],
                              lw=0, markersize=7, label=f)
                   for f in MODEL_FAMILIES]
    ds_handles = [plt.Line2D([0], [0], marker=m, color="#666", lw=0,
                             markersize=7, label=ds)
                  for ds, m in zip(DATASETS, ["o", "s", "D"])]
    axes[0].legend(handles=fam_handles + ds_handles, fontsize=8, frameon=True,
                   ncol=4, loc="lower left", framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        plt.savefig(BASE_DIR / f"generation_timeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: generation_timeline.png / .pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    plot_slopes()
    plot_deltas()
    plot_timeline()
