"""YOLO26 eğitimi (birleşik modeller) — 5070 / herhangi CUDA kutusu.

  python train/train.py --group driver  --data datasets/driver/data.yaml  --model yolo26s.pt --epochs 100
  python train/train.py --group vehicle --data datasets/vehicle/data.yaml --model yolo26s.pt --epochs 120

Eğitim sonunda held-out (val) üzerinde model.val() ölçer ve:
  • weights/driver_actions.pt   (group=driver)
  • weights/vehicle_type.pt     (group=vehicle)
ile yanına <isim>.metrics.json (GERÇEK mAP/P/R/F1 — FTR §4 kanıtı) yazar.

Küçük-set robustluğu: augment (mosaic/flip/hsv) + erken-durdurma (patience).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_NAME = {"driver": "driver_actions", "vehicle": "vehicle_type"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(OUT_NAME))
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s.pt", help="YOLO26 taban (s/m/l)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="5070 8GB: s@640→16, m@640→8")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    name = f"teknofestv3_{args.group}"
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(ROOT / "runs"),
        name=name,
        # küçük-set augment (D-2 §7 gürbüzlük: fps/çözünürlük/ışık/hava çeşitliliği)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.1, erasing=0.2,
        seed=0, deterministic=True, plots=True,
    )

    # Held-out ölçüm (val böl) — GERÇEK sayılar
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val")
    box = metrics.box
    out_pt = ROOT / "weights" / f"{OUT_NAME[args.group]}.pt"
    best = ROOT / "runs" / name / "weights" / "best.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil
        shutil.copy(best, out_pt)
    report = {
        "model": OUT_NAME[args.group],
        "base": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "classes": list(model.names.values()),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50": {model.names[i]: round(float(v), 4) for i, v in enumerate(box.maps)}
        if hasattr(box, "maps") else {},
    }
    (out_pt.with_suffix(".metrics.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nKaydedildi: {out_pt}  +  {out_pt.with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
