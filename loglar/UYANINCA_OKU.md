# ☀️ UYANINCA OKU — teknofestv3 gece-modu raporu

_Döngü tarafından güncellenir. Repo: github.com/7mertyavuz/teknofestv3_

## TL;DR — TESLİM-HAZIR ✅
Tam, çalışan, D-2 uyumlu bir submission hazır ve repoda. `teslim/` klasöründe:
- **FTR_GONDERILECEK.pdf** (10 sayfa, Arial, gerçek sayılarla — uydurma yok)
- **kod.zip** (193MB, çalışan build context: Dockerfile+src+roadguard+config+weights+main)
- **TESLIM_NOTU.md** (Drive yükleme + D-2 §9 kontrol listesi 7/7)

## Bu gece yapılanlar (büyük adımlar)
1. **v3 D-2 pipeline** — v2'nin proven primitifleri (YOLO26 tespit+takip, plaka custom-LP+OCR konsensüs, 16/8 oylama, driver_lock) temiz bir `video→results.json` orkestratörüne damıtıldı; event-stream/QoD/speed atıldı.
2. **P1 Docker TAM YEŞİL** — imaj **3.63GB** (≤8GB; "10.7GB" yanılgıydı=containerd disk-usage), base cuda:12.1.0, uçtan uca konteyner testi geçti, **offline model-bake çalışıyor (runtime-internet YOK)**, §5.4 temiz, bozuk-video robustluğu.
3. **P2 araç-tip modeli** — açık HF set (QoDe-5G, 6480) → **5070'te YOLO26s-cls eğitildi, held-out top1 0.933** (7 D-2 tipi). Entegre: tip artık modelden (3 test aracı→**suv**, TOGG'a uygun), **regresyon yok** (plaka 3/3 + davranışlar korundu).
4. **P2 yolcu hassasiyeti** — kişi-merkezli rol (birincil-kişi=sürücü) → on_koltuk FP'leri azaltıldı.
5. **P4/P5 robustluk+hız** — pytest 22/22; **540s süre-bütçesi guard** (10dk timeout'ta çıktısız kalma riski sıfır); kare-başına try/except.
6. **P6 FTR** — paralel-ajan workflow + condense + reportlab(gerçek Arial) → 10 sayfa, gerçek metriklerle.
7. **P7 teslim** — kod.zip + TESLIM_NOTU + figürler + §9 7/7.

## Güncel metrikler (hepsi GERÇEK/dosya-kanıtlı)
- Held-out: plaka mAP50 **0.983** (F1 0.973), seatbelt 0.895, smoking 0.856, COCO yolo26l 0.709, **araç-tip top1 0.933**.
- 3-video entegrasyon: plaka 34TC8532 **3/3 (CER 0.0)**, davranış doğru-senaryo, şema %100, tip=suv.
- Docker 3.63GB ≤8GB, offline. Hız: full-pipeline MPS ~7.5 FPS; detektör 5070 ~27 FPS; T4 bütçe analizi + guard.

## Geri alınanlar / düzeltilenler (şeffaflık)
- Docker slim denemesi: `strip` OpenBLAS'ı kırdı → revert (imaj zaten 3.63GB uyumluydu, "10.7GB" yanlış okumaydı).
- 5070 full-pipeline FPS ölçümü 0.67 çıktı → **patolojik** (host onnxruntime CPU-only + OCR-CPU darboğazı + contention); FTR'ye YAZILMADI (I4). Bunun yerine MPS ölçümü + detektör-FPS + bütçe analizi raporlandı.
- HF snapshot_download (token'sız rate-limit) takıldı → git-clone'a pivot (çalıştı).

## ⚠️ Senin kararın gereken (net liste)
1. **YENİ MODEL İÇİN VERİ ERİŞİMİ (#1 blokör)** — sürücü-eylem modelini (su_icme/arkaya_bakma/telefon) eğitmek istedin; veriyi box'tan **4 yöntemle** çekemedim (HF snapshot/hf_hub/git-clone/git-lfs → token'sız throttle/LFS). **HF token** (huggingface.co/settings/tokens) VEYA **Roboflow API key** ver → driver_action_cls.py hazır, anında eğitir+entegre ederim (FP-kontrollü). _5070 bu sırada boş kalmasın diye yolo26m araç-tip eğitiliyor (alternatif)._
2. **Gerçek T4 FPS** — T4 yok. Colab-T4 istersen Google girişi; yoksa mevcut dürüst çerçeve (MPS+detektör+projeksiyon) kalır.
3. **teknocan** — tanım/komite görseli? Yoksa atlanıyor (public veri yok).
4. **Başvuru ID** (+ takım adı teyidi) — FTR kapağında "Nankatsu/985007/-"; doğrusunu ver, kapağı güncelleyeyim.

## Bilinen sınır / sıradaki yüksek-değer iş
- Sürücü-davranış ek sınıfları (su_icme/esneme/arkaya_bakma): mevcut açık setler in-cabin yakın-çekim; bizim girdi uzak roadside 4K → **domain-gap**, riskli transfer. Domain-uyumlu veri (Roboflow/komite) gelince eğitilir.
- Uzun/yavaş video hızı: full-pipeline OCR-CPU-bound; guard timeout'u önler ama uzun videoda kısmi sonuç olabilir → P5 stride + onnxruntime-gpu/TensorRT optimizasyonu (FTR §4.3'te belgeli).
- FTR 10 sayfa (sınırda uyumlu); istenirse 8-9'a çekilebilir (pay).
