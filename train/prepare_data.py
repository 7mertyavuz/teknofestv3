"""Veri seti indirme + birleştirme (Roboflow → birleşik YOLO uzayı).

datasets.yaml'daki her seti indirir, ham sınıf adlarını unified_taxonomy ile kanonik
indekslere remap eder ve gruba (driver/vehicle) göre tek bir YOLO veri setinde birleştirir.

Çalıştırma (5070 / herhangi GPU kutusu):
  set ROBOFLOW_API_KEY=...           # (Windows) / export ROBOFLOW_API_KEY=...  (Linux)
  python train/prepare_data.py --group driver  --out datasets/driver
  python train/prepare_data.py --group vehicle --out datasets/vehicle

datasets.yaml biçimi:
  driver:
    - {name: dms1, workspace: ws, project: proj, version: 3, format: yolov11}
  vehicle:
    - {name: vt1, workspace: ws, project: proj, version: 2, format: yolov11}
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_taxonomy import GROUPS, canonical_index  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _read_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _download_roboflow(spec: dict, dest: Path) -> Path:
    """Tek Roboflow setini indirir, indirilen dizinin yolunu döner."""
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY ortam değişkeni gerekli (app.roboflow.com → Settings → API).")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(spec["workspace"]).project(spec["project"])
    ver = proj.version(int(spec["version"]))
    dest.mkdir(parents=True, exist_ok=True)
    ds = ver.download(spec.get("format", "yolov11"), location=str(dest / spec["name"]))
    return Path(ds.location)


def _names_from_data_yaml(ds_dir: Path) -> list[str]:
    dy = ds_dir / "data.yaml"
    if not dy.exists():
        return []
    data = _read_yaml(dy)
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names or [])


def _remap_split(ds_dir: Path, split: str, group: str, raw_names: list[str],
                 out_dir: Path, prefix: str) -> tuple[int, int]:
    """Bir split'i (train/valid/test) remap edip out_dir/{train|val}'a kopyalar.

    Döner: (kopyalanan_görsel, atlanan_görsel). Etiketsiz kalan görsel atlanır
    (yalnız ilgilenmediğimiz sınıfları içerenler → negatif örnek gürültüsü olmasın).
    """
    img_dir = ds_dir / split / "images"
    lbl_dir = ds_dir / split / "labels"
    if not img_dir.exists():
        return 0, 0
    out_split = "val" if split in ("valid", "val", "test") else "train"
    (out_dir / out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / out_split / "labels").mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        new_lines = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                raw_id = int(float(parts[0]))
                if raw_id >= len(raw_names):
                    continue
                ci = canonical_index(group, raw_names[raw_id])
                if ci is None:
                    continue  # ilgilenmediğimiz sınıf → bu bbox atlanır
                new_lines.append(" ".join([str(ci), *parts[1:]]))
        if not new_lines:
            skipped += 1
            continue
        stem = f"{prefix}_{img.stem}"
        shutil.copy(img, out_dir / out_split / "images" / f"{stem}{img.suffix}")
        (out_dir / out_split / "labels" / f"{stem}.txt").write_text("\n".join(new_lines))
        copied += 1
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=str(HERE / "datasets.yaml"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "rf_cache"))
    args = ap.parse_args()

    specs = _read_yaml(Path(args.datasets)).get(args.group, [])
    if not specs:
        raise SystemExit(f"datasets.yaml içinde '{args.group}' grubu boş — slug'ları doldur.")

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cache = Path(args.cache)

    total_c = total_s = 0
    for i, spec in enumerate(specs):
        print(f"[{i+1}/{len(specs)}] indiriliyor: {spec['name']} ({spec['workspace']}/{spec['project']} v{spec['version']})")
        ds_dir = _download_roboflow(spec, cache)
        raw_names = _names_from_data_yaml(ds_dir)
        print(f"    ham sınıflar: {raw_names}")
        for split in ("train", "valid", "test", "val"):
            c, s = _remap_split(ds_dir, split, args.group, raw_names, out_dir, prefix=spec["name"])
            total_c += c
            total_s += s

    names = GROUPS[args.group]["classes"]
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    print(f"\nBİTTİ: {total_c} görsel kopyalandı, {total_s} atlandı → {out_dir}/data.yaml")
    print(f"Sınıflar ({len(names)}): {names}")


if __name__ == "__main__":
    main()
