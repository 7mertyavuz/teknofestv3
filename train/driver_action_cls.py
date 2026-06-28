"""Sürücü-eylem sınıflandırıcı (YOLO26-cls) — D-2 sürücü davranışı boşlukları için.

Veri: HF açık seti gymprathap/Driver-Distracted-Dataset (State Farm tarzı, c0-c9, tek zip).
State Farm → D-2 eşlemesi (FP-güvenli: c0/c5/c8/c9 zengin 'normal' negatifi):
  c0 safe, c5 radio, c8 hair/makeup, c9 passenger → normal
  c1/c3 texting + c2/c4 phone                     → telefonla_konusma
  c6 drinking                                     → su_icme
  c7 reaching behind                              → arkaya_bakma
Sürücü kırpığına (driver_lock ROI) uygulanır. Çıktı: weights/driver_action.pt + metrics.json.

NOT (domain): State Farm in-cabin yakın-çekim; bizim girdi uzak roadside → transfer belirsiz.
Bu yüzden entegrasyonda YÜKSEK eşik + zaman-oylaması (predict.py); 3-video FP kontrolü zorunlu.

Çalıştırma (5070):
  python train/driver_action_cls.py --epochs 30 --imgsz 224 --batch 64 --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "gymprathap/Driver-Distracted-Dataset"
ZIP_NAME = "Distracted-Driver-Detection-Dataset.zip"

CLASS_MAP = {
    "c0": "normal", "c5": "normal", "c8": "normal", "c9": "normal",
    "c1": "telefonla_konusma", "c2": "telefonla_konusma",
    "c3": "telefonla_konusma", "c4": "telefonla_konusma",
    "c6": "su_icme", "c7": "arkaya_bakma",
}
CLASSES = ["normal", "telefonla_konusma", "su_icme", "arkaya_bakma"]


def _find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """c0..c9 adlı dizinleri bul (State Farm yapısı, derinlik bağımsız)."""
    out: dict[str, list[Path]] = defaultdict(list)
    for d in root.rglob("*"):
        if d.is_dir() and d.name.lower() in CLASS_MAP:
            imgs = [p for p in d.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            if imgs:
                out[d.name.lower()].extend(imgs)
    return out


def build(out: Path, val_ratio=0.2, seed=0, local_zip=None) -> dict:
    from huggingface_hub import hf_hub_download

    zip_path = local_zip or hf_hub_download(HF_REPO, ZIP_NAME, repo_type="dataset")
    print(f"[1/3] zip: {zip_path}")
    extract = ROOT / "data" / "_sf_extract"
    if not extract.exists():
        extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract)
    cdirs = _find_class_dirs(extract)
    by_class: dict[str, list[Path]] = defaultdict(list)
    for cN, imgs in cdirs.items():
        by_class[CLASS_MAP[cN]].extend(imgs)
    print("[2/3] ham (D-2 eşlenmiş):", {k: len(v) for k, v in by_class.items()})
    if not by_class:
        raise SystemExit("c0-c9 sınıf dizini bulunamadı — zip yapısını kontrol et.")

    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": Counter(), "val": Counter()}
    train_per: dict[str, list[Path]] = {}
    for cls, imgs in by_class.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        # büyük setlerde sınıf başına tavan (dengeli + hızlı): 3000
        imgs = imgs[:3000]
        n_val = max(1, int(len(imgs) * val_ratio))
        val, tr = imgs[:n_val], imgs[n_val:]
        train_per[cls] = tr
        for split, lst in (("train", tr), ("val", val)):
            dd = out / split / cls
            dd.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(lst):
                shutil.copy(p, dd / f"{cls}_{i}{p.suffix}")
                counts[split][cls] += 1
    # denge: azınlık train sınıflarını en büyüğe oversample
    target = max(len(v) for v in train_per.values())
    for cls, tr in train_per.items():
        dd = out / "train" / cls
        j = 0
        while counts["train"][cls] < target and tr:
            p = tr[j % len(tr)]
            shutil.copy(p, dd / f"{cls}_os_{j}{p.suffix}")
            counts["train"][cls] += 1
            j += 1
    print("[3/3] train:", dict(counts["train"]), "| val:", dict(counts["val"]))
    return {"train": dict(counts["train"]), "val": dict(counts["val"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--out", default=str(ROOT / "data" / "driver_cls"))
    ap.add_argument("--local_zip", default=None)
    a = ap.parse_args()

    dist = build(Path(a.out), local_zip=a.local_zip)
    from ultralytics import YOLO
    try:
        model = YOLO(a.model)
    except Exception as e:
        print(f"{a.model} yok ({e}) → yolo11s-cls.pt")
        model = YOLO("yolo11s-cls.pt")
    name = "teknofestv3_driver_action"
    model.train(data=a.out, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(ROOT / "runs"), name=name, cache=True,  # cache=True → GPU-bound (hızlı)
                hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8, translate=0.1, scale=0.5,
                fliplr=0.5, erasing=0.3, seed=0, deterministic=True, plots=True)
    metrics = model.val(data=a.out, imgsz=a.imgsz, device=a.device, split="val")
    out_pt = ROOT / "weights" / "driver_action.pt"
    best = ROOT / "runs" / name / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, out_pt)
    report = {"model": "driver_action", "task": "classify", "base": a.model,
              "epochs": a.epochs, "imgsz": a.imgsz, "classes": list(model.names.values()),
              "top1": round(float(getattr(metrics, "top1", 0.0)), 4),
              "top5": round(float(getattr(metrics, "top5", 0.0)), 4),
              "train_dist": dist["train"], "val_dist": dist["val"], "source": HF_REPO,
              "note": "State Farm in-cabin → roadside transfer belirsiz; entegrasyonda yüksek esik+oylama"}
    (out_pt.with_suffix(".metrics.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
