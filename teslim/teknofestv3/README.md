# teknofestv3 — TEKNOFEST 2026 · 5G & Yapay Zekâ ile Akıllı Yol Güvenliği (FTR)

Yol kenarı kamera videosundan **tek geçişte** araç bilgisi (tip/plaka/renk) ve yol
güvenliği tespitleri (sürücü eylemleri, kabin nesneleri, yolcular) üretip
**D-2 şemasına birebir uyumlu** `results.json` yazan, **Tesla T4** için Dockerize
çıkarım paketi.

```
/app/data/input/video.mp4  ──►  main.py  ──►  /app/data/output/results.json
```

## Mimari (kuşbakışı)

```
video.mp4
  │  cv2 kare akışı (fps-bağımsız)
  ▼
[Stage-1] YOLO26 araç + kişi tespiti + ByteTrack takip ─┐
  │  alan-ağırlıklı sınıf oyu, min-track kapısı          │ driver_lock
  ▼                                                       ▼ (sürücü/yolcu)
[Stage-2a] Sürücü durumu (YOLO26-pose hibrit + custom_smoking/seatbelt)
           16/8 zaman-oylaması → sigara/telefon/kemer/slalom
[Stage-2b] Plaka: custom_license_plate (mAP50 0.983) sıkı kırpma
           → fast-plate-ocr / EasyOCR + TR-normalizasyon + ağırlıklı oy konsensüsü
[Attrs]    Araç tipi (7-sınıf model/heuristik) + renk (HSV baskın-renk)
  │
  ▼  ID-merkezli birikim → ana araç + zaman damgalı olay epizotları
D-2 results.json  (şema doğrulayıcıdan geçer)
```

Detaylar: `src/predict.py` (orkestratör), `src/d2_labels.py` (D-2 sözleşmesi + doğrulayıcı),
`src/vehicle_attrs.py` (tip/renk), `roadguard/` (tespit/takip/plaka/sürücü-durum primitifleri).

## D-2 uyum

| Kural | Durum |
|---|---|
| `/app/data/input/video.mp4` oku → `/app/data/output/results.json` yaz | ✅ `main.py` (sabit yol) |
| Şema anahtarları birebir (`arac_bilgisi{tip,plaka,renk,confidence_score}`, `tespitler[{zaman_saniye,kategori,etiket,confidence_score}]`) | ✅ `d2_labels.validate_results` (CI testli) |
| Etiketler ASCII + küçük harf (Türkçe karakter yok) | ✅ beyaz-liste + `to_ascii_lower` |
| Plaka TR-regex (01-81) ya da `"tespit edilemedi"` | ✅ `normalize_plate` |
| `confidence_score ∈ [0,1]` float | ✅ `clamp_conf` |
| base `nvidia/cuda:12.1.0-base-ubuntu22.04`, GPU=cuda | ✅ `Dockerfile` |
| İmaj ≤8GB, çalışma ≤10dk, runtime internet YOK | ✅ torch cu121 + offline model-bake |
| §5.4: ortam/IP/hostname/env tespitiyle davranış değiştirme YOK | ✅ tek davranış, sabit yol |
| Bozuk/eksik video'da çökme yok | ✅ kare-başına + global try/except, fallback results.json |

## Çalıştırma (D-2 hakem akışı)

```bash
docker build -t teknofest/teknofestv3:latest .
docker run --rm --gpus all \
  -v /yol/video.mp4:/app/data/input/video.mp4 \
  -v /yol/cikti:/app/data/output \
  teknofest/teknofestv3:latest
```

## Yerel geliştirme / test

```bash
pip install -r requirements.txt   # + torch (cu121/cpu/mps)
pytest tests/                      # D-2 şema sözleşmesi testleri
# tek video:
python -c "from src.predict import run_inference; import json; print(json.dumps(run_inference('video_1.mp4','weights'),ensure_ascii=False,indent=2))"
```

## Modeller (`weights/`, YOLO26 tabanlı)

| Ağırlık | Görev | Held-out |
|---|---|---|
| `yolo26l.pt` | Stage-1 araç/kişi/nesne (stok COCO) | COCO mAP50 0.709 |
| `yolo26l-pose.pt` | Sürücü pose (kişi keypoint) | stok |
| `custom_license_plate.pt` | Plaka tespiti (sıkı kırpma) | mAP50 **0.983** |
| `custom_smoking.pt` | Sigara tespiti | mAP50 **0.856** |
| `custom_seatbelt.pt` | Emniyet kemeri | mAP50 **0.895** |

Araç-tip (7-sınıf) ve birleşik sürücü-eylem/nesne modelleri eğitim aşamasında
(`train/`) eklenir; mevcut değilse sistem heuristik + dürüst düşük-güvene düşer
(asla uydurma değer üretmez).
