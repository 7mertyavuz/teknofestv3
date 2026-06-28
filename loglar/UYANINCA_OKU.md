# ☀️ UYANINCA OKU — teknofestv3 gece-modu raporu

_Bu dosya döngü tarafından sürekli güncellenir. En güncel hâli görürsün._

## TL;DR (tek bakış)
- **Çalışan submission repoda hazır** (en kötü senaryo: bu teslim edilebilir). 3-video geçerli, plaka 34TC8532 3/3, senaryolar 3/3 doğru.
- Gece boyunca yapılanlar aşağıda; geri alınanlar şeffaf listelendi.

## Bu gece yapılanlar (commit + metrik etkisi)
- (Iter 0) v3 D-2 paketi + log altyapısı kuruldu. _Sonraki iterasyonlar buraya eklenecek._

## Güncel durum
- Uyumluluk: şema ✅ (pytest 10/10), Docker ⏳ (build doğrulanıyor)
- Baseline: bkz STATE.md tablosu (tip/renk/plaka/tespit per-video)
- Held-out per-etiket P/R/F1: _eğitim sonrası eklenecek_
- T4 FPS/süre: _ölçülecek_

## Geri alınanlar (şeffaflık)
- _henüz yok_

## ⚠️ Senin kararın gerekenler
1. **Roboflow API key** (varsa): app.roboflow.com → Settings → API. Verirsen çok daha zengin/etiketli veri → daha iyi modeller. Yoksa açık setlerle devam ediyorum.
2. **Gerçek T4 FPS**: Elde T4 yok. Colab-T4'te ölçmek istersen Google girişi gerekir; istersen ben 5070/CPU'da ölçüp "T4-projeksiyon" diye dürüst etiketlerim.
3. **teknocan**: nedir? Komite görseli/tanımı var mı? Yoksa dürüst düşük-güven bırakıyorum.

## Sıradaki en yüksek-değerli 3 iş
1. Docker imaj doğrula + boyut/süre ölç (compliance %100).
2. Açık-veri → birleşik YOLO26 (araç-tip + sürücü-eylem) eğit, held-out kanıtla.
3. FTR taslağı gerçek metriklerle.
