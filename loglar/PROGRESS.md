# teknofestv3 — PROGRESS (otonom gece-modu iterasyon günlüğü)

Format: her iterasyon → [saat] SEÇİM (Pn) · NE · SONUÇ · KARAR (commit/revert) · METRİK etkisi

---

## Iter 0 — 2026-06-28 gece, başlangıç
- **Durum kuruldu**: v3 D-2 paketi repoda (github.com/7mertyavuz/teknofestv3), pytest 10/10, 3-video baseline geçerli (plaka 34TC8532 3/3, senaryolar 3/3 doğru).
- **Altyapı**: 5070 CUDA 12.8 eğitime hazır; win-WSL native amd64 Docker hazır (build koşuyor); Codex review uygulandı (main.py env-override kaldırıldı → §5.4 optiği; predict.py kare-başına dayanıklılık).
- **Eğitim scaffolding**: train/{unified_taxonomy,prepare_data,train}.py + datasets.yaml.
- **Karar**: v2 cache'li setler KULLANILMAYACAK (kullanıcı direktifi). Roboflow key yok → Roboflow-dışı açık setler (agy araştırıyor). Aug+balance zorunlu.
- **Sıradaki**: P1 Docker doğrula → P2/P3 açık-veri + birleşik YOLO26 eğitim.

## Iter 1 — P4 test kapsamı
- **NE**: dayanıklılık + saf-yardımcı testleri eklendi (epizot çökeltme, koltuk geometrisi, fallback/yazıcı roundtrip, ASCII-disk, validator kenar durumları).
- **SONUÇ**: pytest 10→22 yeşil. **KARAR**: commit f051baf + push.

## Iter 2 — P1 Docker uyumluluk + P2 araç-tip eğitimi
- **ÖLÇ**: ilk Docker build başarılı (warmup+offline-bake OK) AMA **imaj 10.7GB > 8GB → İHLAL**.
- **NE (P1)**: Dockerfile slimleme — kurulum+strip(.so)+temizlik(test/pyc/.a/.h/duplike-opencv) TEK RUN'da (layer additivity). binutils eklendi. Build warmup'ı strip'i kendi doğrular (kırılırsa build patlar).
- **NE (P2)**: veri darboğazı → Roboflow yok; agy takıldı (boş). **Açık set bulundu**: HF `QoDe-5G/raw-car-classification-iHasibi` (6480 görsel, SEDAN/SUV/Hatchback/Pickup/Truck/Bus/MPV → D-2 7-tip eşlemesi, no-key). YOLO26-cls araç-tip eğitimi 5070'te başlatıldı (oversample-denge + augment).
- **KARAR**: commit 009fda9 + push. İki rebuild/train arka planda; sonuç beklenecek.
- **DURUM**: ⏳ Docker rebuild (boyut ölçülecek) + ⏳ araç-tip eğitim (held-out top1 ölçülecek).

## Iter 3 — P1 Docker "ihlali" YANILGI çıktı + yolcu hassasiyeti (KIRMIZI/MAVİ)
- **KIRMIZI**: "imaj 10.7GB" diye slimledim; slim build warmup'ta PATLADI — `strip --strip-unneeded` OpenBLAS .so'yu bozdu (ELF page-align → numpy import fail).
- **MAVİ/ÖLÇÜM**: Konteynerde ölçtüm → **gerçek imaj boyutu `docker image inspect .Size` = 3.63GB** (10.7GB containerd "disk usage"=build-cache, imaj DEĞİL). **İmaj ZATEN ≤8GB UYUMLU.** Ayrıca: strip-debug sıfır kazandırıyor (libs zaten stripli); nccl+cupti torch import'ta ZORUNLU (kaldırılamaz).
- **KARAR (I5)**: slim değişikliği gereksiz + kırıyordu → **Dockerfile orijinale REVERT**. Mevcut 3.63GB imaj geçerli, kullanılıyor.
- **EK (P2 hassasiyet)**: baseline'daki `on_koltuk` yanlış-pozitifleri için yolcu epizotuna **min-kare kapısı (≥8)** eklendi (geçici tek-kare yolcu elenir). pytest 22/22 yeşil.
- **METRİK**: Docker uyumluluk ✅ (3.63GB, build+warmup OK). Sıradaki: imajı bir videoyla çalıştırıp results.json doğrula (P1 kapanış).

## Iter 4 — P1 KAPANDI (Docker uçtan uca) + araç-tip veri pivotu
- **P1 DOĞRULAMA**: imaj bir klipte (CPU, WSL) çalıştırıldı → **geçerli results.json üretti** (tip=sedan, renk, slalom@2.5s; downscale klip → plaka 'tespit edilemedi' = doğru). **fast-plate-ocr modeli baked-cache'den yüklendi ("Skipping download... already exists") → OFFLINE BAKING + runtime-internetsiz ÇALIŞIYOR.** P1 (build+run+offline+şema+≤8GB) **TAM YEŞİL**.
- **P2 veri (KIRMIZI/MAVİ)**: `snapshot_download` 6480 dosya token'sız rate-limit'te takıldı (cache boş, 25dk) → python kill. **PIVOT**: HF dataset'i **git clone** ile çekiliyor (tek bağlantı, per-file throttle yok). vehicle_type_cls.py'ye `--local_src` eklendi (yerel klonu kullan).
- **KARAR**: commit+push; git clone arka planda; bitince --local_src ile eğitim re-run.

## Iter 5 — PARALEL BATCH (FTR + teslim paketi + figürler + araç-tip eğitim)
- **P6 FTR**: paralel-ajan workflow (5 bölüm + editör) → 27K markdown; condense-ajan → 16K (sayılar+6 tablo korundu); reportlab+gerçek-Arial render → **FTR_GONDERILECEK.pdf 10 sayfa (≤10 ✓)**. Figürler: Şekil1 mimari + Şekil2 mAP (matplotlib).
- **P7 teslim**: kod.zip (160MB, 98 dosya, tekrar-üretilebilir build script) + TESLIM_NOTU.md (Drive + D-2 §9 **7/7 DOĞRULANDI**).
- **P2 araç-tip**: git-clone veri (6483) → 5070'te YOLO26s-cls (train 5222/val 822/7 sınıf) — epoch 24/40 aktif.
- **P2 yolcu**: kişi-merkezli rol (birincil-kişi=sürücü) → on_koltuk FP video_2/3 temizlendi, video_1'de 1 kaldı (≥8-kare ikincil kişi, 0.5 dürüst). pytest 22/22.
- **KARAR**: hepsi commit+push. Eğitim bitince: vehicle_type.pt entegre + re-baseline + 5070 FPS-bench + FTR top1 güncelle.

## Iter 6 — Araç-tip eğitim TAMAM + ENTEGRE (P2 doğruluk kazancı)
- **EĞİTİM**: YOLO26s-cls, results.csv: epoch31 **top1 0.933** / top5 0.999 (epoch28'den plato). CPU/DataLoader-bound (~2 it/s; GPU %0-1 — ders: cache=True/küçük girdi). Plato görülünce epoch35'te durdurup best.pt kullandım (boşa beklemedim).
- **ENTEGRE**: best.pt→weights/vehicle_type.pt (scp). 3-video re-baseline: **tip artık modelden → 3 araç da "suv"** (TOGG T10X SUV; heuristik "sedan"dan muhtemelen daha doğru). **REGRESYON YOK**: plaka 34TC8532 3/3, davranışlar (sigara/telefon/slalom) korundu, şema geçerli, cs 0.87-0.92'ye yükseldi.
- **FTR**: top1 0.933 işlendi (Tablo1 + §3.3 + özet; "ölçülüyor" kaldırıldı; classification notu eklendi). PDF 10sf. kod.zip yenilendi (193MB). pytest 22/22. **KARAR**: commit dc02688 + push.
- **NOT (5070 kullanımı)**: araç-tip body-tip uzaktan görünür → transfer iyi. Sürücü-davranış (su_icme/esneme) setleri in-cabin yakın-çekim; bizim girdi uzak roadside 4K → domain-gap, riskli transfer (I4) → domain-uyumlu komite/Roboflow verisi gerekli (sabah notu). 5070: araç-tip + FPS-bench ile kullanıldı.
