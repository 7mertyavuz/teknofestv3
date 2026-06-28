# teknofestv3 — PROGRESS (otonom gece-modu iterasyon günlüğü)

Format: her iterasyon → [saat] SEÇİM (Pn) · NE · SONUÇ · KARAR (commit/revert) · METRİK etkisi

---

## Iter 0 — 2026-06-28 gece, başlangıç
- **Durum kuruldu**: v3 D-2 paketi repoda (github.com/7mertyavuz/teknofestv3), pytest 10/10, 3-video baseline geçerli (plaka 34TC8532 3/3, senaryolar 3/3 doğru).
- **Altyapı**: 5070 CUDA 12.8 eğitime hazır; win-WSL native amd64 Docker hazır (build koşuyor); Codex review uygulandı (main.py env-override kaldırıldı → §5.4 optiği; predict.py kare-başına dayanıklılık).
- **Eğitim scaffolding**: train/{unified_taxonomy,prepare_data,train}.py + datasets.yaml.
- **Karar**: v2 cache'li setler KULLANILMAYACAK (kullanıcı direktifi). Roboflow key yok → Roboflow-dışı açık setler (agy araştırıyor). Aug+balance zorunlu.
- **Sıradaki**: P1 Docker doğrula → P2/P3 açık-veri + birleşik YOLO26 eğitim.
