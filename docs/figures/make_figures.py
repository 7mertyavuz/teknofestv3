"""FTR figürleri: Şekil 1 (sistem mimarisi) + Şekil 2 (held-out mAP karşılaştırma).

matplotlib ile üretilir (harici araç gerekmez). Çıktı: docs/figures/*.png (FTR PDF'e gömülür).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor="#222", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color="#444"))


def architecture():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    blue, green, orange, gray, yellow = "#cfe3ff", "#d6f0d2", "#ffe2c2", "#e8e8e8", "#fff3bf"

    _box(ax, 0.2, 2.6, 1.5, 0.9, "video.mp4\n(cv2, fps-bagimsiz)", gray)
    _box(ax, 2.1, 2.6, 1.9, 0.9, "Stage-1\nYOLO26l + ByteTrack\narac/kisi tespit+takip", blue)
    _box(ax, 4.4, 4.5, 1.9, 0.9, "Stage-2a Surucu Durumu\nYOLO26-pose + custom_smoking\n16/8 zaman-oyu", green)
    _box(ax, 4.4, 3.1, 1.9, 0.9, "Stage-2b Plaka\ncustom_LP -> fast-plate-ocr\nTR-normalize + oy konsensus", green)
    _box(ax, 4.4, 1.7, 1.9, 0.9, "Arac Ozellikleri\nYOLO26-cls tip + HSV renk", green)
    _box(ax, 4.4, 0.4, 1.9, 0.8, "DriverLock\nsurucu/yolcu atama", yellow)
    _box(ax, 6.7, 2.6, 1.6, 0.9, "ID-merkezli\nbirikim + epizot\nfuzyonu", orange)
    _box(ax, 8.5, 2.6, 1.4, 0.9, "results.json\n(D-2 sema\ndogrulanir)", gray)

    _arrow(ax, 1.7, 3.05, 2.1, 3.05)
    _arrow(ax, 4.0, 3.2, 4.4, 4.9)
    _arrow(ax, 4.0, 3.05, 4.4, 3.55)
    _arrow(ax, 4.0, 2.9, 4.4, 2.1)
    _arrow(ax, 4.0, 2.8, 4.4, 0.8)
    _arrow(ax, 6.3, 4.9, 6.7, 3.4)
    _arrow(ax, 6.3, 3.55, 6.7, 3.2)
    _arrow(ax, 6.3, 2.1, 6.7, 2.9)
    _arrow(ax, 8.3, 3.05, 8.5, 3.05)
    ax.set_title("Sekil 1: teknofestv3 Sistem Mimarisi (video → results.json)", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "mimari.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_chart():
    models = ["license_plate\n(YOLO26s)", "seatbelt\n(YOLO26s)", "smoking\n(YOLO26s)", "yolo26l\n(COCO val)"]
    vals = [0.983, 0.895, 0.856, 0.709]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(models, vals, color=["#2a9d8f", "#457b9d", "#e9c46a", "#adb5bd"], edgecolor="#222")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("mAP@0.5 (held-out)")
    ax.set_title("Sekil 2: Held-out mAP@0.5 (model.val, val≠test)", fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "map_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    map_chart()
    print("figures yazildi:", os.listdir(HERE))
