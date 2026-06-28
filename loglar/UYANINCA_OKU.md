# ☀️ UYANINCA OKU — teknofestv3 gece-modu raporu

_Döngü tarafından sürekli güncellenir._

## TL;DR
- **Teslim-hazır bir submission VAR** (en kötü senaryo bu gider): Docker imajı **3.63GB ≤8GB**, uçtan uca doğrulandı (offline model-bake çalışıyor, runtime-internet YOK), 3-video geçerli (plaka 34TC8532 3/3, sigara/telefon/slalom doğru), **D-2 §9 7/7**, **FTR_GONDERILECEK.pdf 10 sayfa** + kod.zip + TESLIM_NOTU hazır (`teslim/`).
- Repo güncel: github.com/7mertyavuz/teknofestv3.

## Bu gece yapılanlar (özet)
- v3 D-2 pipeline (roadguard primitifleri damıtıldı; event-stream/QoD/speed atıldı) → temiz video→results.json.
- **P1 Docker**: 3.63GB (10.7GB "ihlali" yanılgıydı=containerd disk-usage), build+run+offline+§5.4+robustluk **TAM YEŞİL**, uçtan uca konteyner testi geçti.
- **P4**: pytest 22/22 (şema/plaka/ASCII/epizot/koltuk/robustluk).
- **P2 araç-tip**: HF açık set (QoDe-5G, 6480) → 5070'te YOLO26s-cls eğitiliyor (7 D-2 tipi, oversample-denge) — "hep sedan"ı çözer.
- **P2 yolcu**: rol-salınım on_koltuk FP'si kişi-merkezli mantıkla azaltıldı.
- **P6 FTR**: paralel-ajan + condense + reportlab(Arial) → 10 sayfa, GERÇEK sayılarla (uydurma yok).
- **P7 teslim**: kod.zip + TESLIM_NOTU(§9 7/7) + figürler.

## Güncel metrikler
- Held-out (model.val): license_plate mAP50 **0.983**, seatbelt 0.895, smoking 0.856; COCO yolo26l 0.709.
- 3-video: plaka 3/3 (CER 0.0), araç-sınıfı %100, davranış doğru-senaryo, şema %100.
- **araç-tip top1: 0.933** (YOLO26s-cls, 7 D-2 tipi, 822 görsel held-out) — ENTEGRE: tip artık modelden (3 test aracı→suv), regresyon yok (plaka 3/3 + davranış korundu).
- Docker: 3.63GB, T4/cu121, offline. FPS: 5070 full-pipeline ölçümü koşuyor (FTR'ye gerçek sayı); T4 projeksiyon ~27 (T4 elde yok).

## ⚠️ Senin kararın gerekenler
1. **Roboflow API key** (varsa) → çok daha zengin veri = daha iyi modeller. Yoksa açık setlerle devam (şu an öyle).
2. **Gerçek T4 FPS**: T4 yok. 5070'te ölçüp "T4-projeksiyon" etiketliyorum; Colab-T4 istersen Google girişi gerekir.
3. **teknocan**: tanım/veri yok → dürüst atlandı. Komite görseli varsa ekleyebilirim.
4. **Takım/Başvuru ID**: FTR kapağında "Nankatsu / 985007 / -" var; Başvuru ID'yi ver, güncelleyeyim.

## Sıradaki (otomatik)
1. araç-tip eğitimi bitince → vehicle_type.pt entegre + 3-video re-baseline (regresyonsuzsa commit) + held-out top1 → FTR.
2. 5070'te v3 full-pipeline FPS ölç (FTR §4 gerçek sayı).
3. FTR'yi 9 sayfaya çekip pay bırak; gerekirse driver-state (su_icme/esneme) için veri+eğitim (5070).
