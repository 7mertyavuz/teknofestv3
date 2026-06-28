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
