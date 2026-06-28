"""Araç-tip sınıflandırıcı (YOLO26-cls) — D-2 arac_bilgisi.tip için.

Veri: HF açık seti QoDe-5G/raw-car-classification-iHasibi (no-key, 6480 görsel + type CSV).
Bangladesh-ağırlıklı genel araç fotoğrafları; gövde-tipi transfer edilebilir. Sınıf eşlemesi:
  SEDAN→sedan, SUV→suv, Hatchback→hatchback, Pickup→pickup, Truck→kamyon, Bus→minibus, MPV→panelvan
  (Bike/Easy-Bike/CNG → D-2 dışı, atılır). DENGE: train'de azınlık sınıflar oversample edilir.

Çıktı: weights/vehicle_type.pt (cls) + vehicle_type.metrics.json (GERÇEK held-out top1/top5).
Pipeline (src/vehicle_attrs.VehicleTypeClassifier) bu .pt'yi otomatik kullanır (probs head).

Çalıştırma (5070 / CUDA kutusu):
  pip install huggingface_hub
  python train/vehicle_type_cls.py --epochs 40 --imgsz 224 --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "QoDe-5G/raw-car-classification-iHasibi"

# Kaynak 'Type' (CSV) → D-2 tip. Listede olmayanlar (Bike/Easy-Bike/CNG) ATILIR.
TYPE_MAP = {
    "SEDAN": "sedan", "SUV": "suv", "Hatchback": "hatchback",
    "Pickup": "pickup", "Truck": "kamyon", "Bus": "minibus", "MPV": "panelvan",
}


def build_cls_dataset(out: Path, val_ratio=0.2, seed=0, oversample=True, local_src=None) -> dict:
    if local_src:
        src = Path(local_src)
        print(f"[1/3] yerel kaynak kullanılıyor: {src}")
    else:
        from huggingface_hub import snapshot_download

        print(f"[1/3] HF indiriliyor: {HF_REPO}")
        src = Path(snapshot_download(HF_REPO, repo_type="dataset"))
    csv_path = src / "type-labels.csv"
    train_dir = src / "Train"

    rows = list(csv.reader(open(csv_path, encoding="utf-8-sig")))[1:]
    by_class: dict[str, list[Path]] = defaultdict(list)
    for r in rows:
        if len(r) < 2:
            continue
        img_id, raw_type = r[0].strip(), r[1].strip()
        d2 = TYPE_MAP.get(raw_type)
        if not d2:
            continue
        img = train_dir / f"{img_id}.jpg"
        if img.exists():
            by_class[d2].append(img)

    print("[2/3] sınıf dağılımı (ham):", {k: len(v) for k, v in by_class.items()})
    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": Counter(), "val": Counter()}
    train_per_class: dict[str, list[Path]] = {}
    for cls, imgs in by_class.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val, tr = imgs[:n_val], imgs[n_val:]
        train_per_class[cls] = tr
        for split, lst in (("train", tr), ("val", val)):
            dd = out / split / cls
            dd.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(lst):
                shutil.copy(p, dd / f"{cls}_{i}{p.suffix}")
                counts[split][cls] += 1

    # DENGE: train'de azınlık sınıfları en-büyük sınıfa yakınsa oversample (kopya).
    if oversample and train_per_class:
        target = max(len(v) for v in train_per_class.values())
        for cls, tr in train_per_class.items():
            dd = out / "train" / cls
            have = counts["train"][cls]
            need = target - have
            j = 0
            while need > 0 and tr:
                p = tr[j % len(tr)]
                shutil.copy(p, dd / f"{cls}_os_{j}{p.suffix}")
                counts["train"][cls] += 1
                need -= 1
                j += 1
    print("[3/3] train:", dict(counts["train"]), "| val:", dict(counts["val"]))
    return {"train": dict(counts["train"]), "val": dict(counts["val"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--out", default=str(ROOT / "data" / "vehicle_cls"))
    ap.add_argument("--local_src", default=None, help="HF yerine yerel klon dizini (type-labels.csv + Train/)")
    args = ap.parse_args()

    dist = build_cls_dataset(Path(args.out), local_src=args.local_src)

    from ultralytics import YOLO

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"{args.model} yüklenemedi ({e}) → yolo11s-cls.pt'ye düşülüyor")
        model = YOLO("yolo11s-cls.pt")

    name = "teknofestv3_vehicle_type"
    model.train(
        data=args.out, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(ROOT / "runs"), name=name,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8, translate=0.1, scale=0.5,
        fliplr=0.5, erasing=0.3, seed=0, deterministic=True, plots=True,
    )
    metrics = model.val(data=args.out, imgsz=args.imgsz, device=args.device, split="val")
    out_pt = ROOT / "weights" / "vehicle_type.pt"
    best = ROOT / "runs" / name / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, out_pt)
    report = {
        "model": "vehicle_type", "task": "classify", "base": args.model,
        "epochs": args.epochs, "imgsz": args.imgsz,
        "classes": list(model.names.values()),
        "top1": round(float(getattr(metrics, "top1", 0.0)), 4),
        "top5": round(float(getattr(metrics, "top5", 0.0)), 4),
        "train_dist": dist["train"], "val_dist": dist["val"],
        "source": HF_REPO, "note": "Bangladesh-agirlikli genel arac seti; govde-tipi transfer",
    }
    (out_pt.with_suffix(".metrics.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nKaydedildi: {out_pt}")


if __name__ == "__main__":
    main()
