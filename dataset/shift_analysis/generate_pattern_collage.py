#!/usr/bin/env python3
"""
Generate a 2×2 collage of representative images for the 4 regression patterns.

(A) Tiny figure in manuscript page   — Paul → John
(B) Landscape-dominant hermit scene  — Antony Abbot → Jerome
(C) Multi-saint composition          — M. Magdalene → Catherine
(D) Greyscale / monochrome print     — Antony Abbot → Francis

Output: dataset/shift_analysis/regression_pattern_collage.pdf / .png
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path(__file__).parent.parent.parent
IMG_DIR = BASE_DIR / "dataset" / "ICONCLASS" / "JPEGImages"
OUTPUT_DIR = Path(__file__).parent

PANELS = [
    {
        "letter": "(A)",
        "image": "biblia_sacra_20110822121.jpg",
        "caption": "Manuscript page with small historiated figure\nTrue: St. Paul — Pred: St. John",
    },
    {
        "letter": "(B)",
        "image": "IIHIM_1195649509.jpg",
        "caption": "Landscape-dominant hermit scene\nTrue: St. Antony Abbot — Pred: St. Jerome",
    },
    {
        "letter": "(C)",
        "image": "IIHIM_RIJKS_2033920572.jpg",
        "caption": "Multi-saint altarpiece composition\nTrue: Mary Magdalene — Pred: St. Catherine",
    },
    {
        "letter": "(D)",
        "image": "IIHIM_221633182.jpg",
        "caption": "Greyscale / monochrome print\nTrue: St. Antony Abbot — Pred: St. Francis",
    },
]


def main():
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.08,
                           left=0.03, right=0.97, top=0.93, bottom=0.02)

    for idx, panel in enumerate(PANELS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        img_path = IMG_DIR / panel["image"]
        img = Image.open(img_path).convert("RGB")

        ax.imshow(img)
        ax.set_axis_off()

        # Letter label in top-left corner
        ax.text(0.02, 0.98, panel["letter"],
                transform=ax.transAxes, fontsize=16, fontweight="bold",
                fontfamily="serif", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#999", alpha=0.9))

        # Caption below image
        ax.text(0.5, -0.04, panel["caption"],
                transform=ax.transAxes, fontsize=9, fontfamily="serif",
                ha="center", va="top", linespacing=1.4)

    fig.suptitle("Visual Patterns in GPT-5.2 Misclassification Cases (ICONCLASS)",
                 fontsize=14, fontweight="bold", fontfamily="serif", y=0.97)

    out_pdf = OUTPUT_DIR / "regression_pattern_collage.pdf"
    out_png = OUTPUT_DIR / "regression_pattern_collage.png"
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Collage saved → {out_pdf}")
    print(f"Collage saved → {out_png}")


if __name__ == "__main__":
    main()
