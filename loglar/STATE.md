# teknofestv3 — STATE (anlık durum)

_Son güncelleme: gece-modu iterasyon 0 (başlangıç ölçümü), 2026-06-28_

## Uyumluluk (D-2) — P1
| Madde | Durum |
|---|---|
| Şema validatörü (anahtar/etiket/ASCII/plaka/conf) | ✅ pytest 10/10 |
| 3-video → geçerli results.json | ✅ (v1/v2/v3 VALIDATION OK) |
| Docker imajı (T4/cu121) build + run | ✅ **3.63GB**, uçtan uca doğrulandı (klipte geçerli results.json) |
| İmaj ≤8GB / runtime-internetsiz | ✅ 3.63GB ≤8GB; offline model-bake çalışıyor ("Skipping download... already exists") |
| Çalışma ≤10dk | ⏳ T4'te ölçülemiyor (bizde T4 yok); CPU klip 75 kare/82s — T4 GPU'da bütçe içi beklenir |
| §5.4 ortam-tespiti yok | ✅ sabit yol, tek davranış |
| Bozuk/eksik video çökme yok | ✅ kare-başına + global try/except + fallback |

## Baseline metrikler (yerel MPS, full 3-video)
| Video | tip | renk | plaka | cs | tespitler | süre |
|---|---|---|---|---|---|---|
| video_1 | sedan | siyah | 34TC8532 ✓ | 0.77 | sigara_icme@2.6 ✓, on_koltuk@3.8 | 56s |
| video_2 | sedan | siyah | 34TC8532 ✓ | 0.83 | telefonla_konusma@4.5 ✓, on_koltuk×2 | 47s |
| video_3 | sedan | siyah | 34TC8532 ✓ | 0.82 | slalom@2.2 ✓, on_koltuk@1.9 | 57s |

GT (data/samples): plaka 34TC8532 (3/3 ✓), senaryolar sigara/telefon/slalom (3/3 ✓).

## Yetenek kapsama (D-2 etiket → durum)
- ✅ GERÇEK MODEL: plaka (custom_LP mAP50 0.983), sigara_icme (0.856), emniyet_kemeri (0.895)
- ✅ TÜRETİLMİŞ: slalom (yörünge), renk (HSV)
- 🟡 HEURİSTİK/STOK: telefonla_konusma (COCO+pose), bilgisayar (COCO laptop), yolcular (geometri), araç-tip (→sedan fallback)
- ❌ MODEL YOK: araç-tip 7-sınıf (eğitilecek), esneme, su_icme, arkaya_bakma, etrafa_bakinma, teknocan

## İnsan-kararı/engel
- Roboflow API key VERİLMEDİ + MCP propagasyon yok → açık (Roboflow-dışı, key'siz) setlere düşülüyor (kullanıcı onayladı). v2 cache'li setler KULLANILMIYOR (kullanıcı: yetersiz/format yok).
- Gerçek T4 FPS: elde T4 yok (5070 Blackwell). FPS açık donanımda ölçülüp "T4-projeksiyon" etiketlenecek ya da Colab-T4 (giriş gerekirse insan).
- teknocan: tanım/veri yok → dürüst düşük-güven/atla.

## Sıradaki en yüksek-değerli 3 iş
1. P1: Docker build doğrula + imaj boyutu/süre ölç (compliance %100).
2. P2/P3: açık setlerden veri → birleşik YOLO26 (araç-tip + sürücü-eylem incl. esneme/su_icme), augment+balance, 5070'te eğit, held-out ölç.
3. P6: FTR taslağı gerçek metriklerle (custom metrics.json + baseline + ölçülen FPS).

---
## GÜNCEL (gece sonu — bu bölüm üsttekileri geçersiz kılar)
**Uyumluluk P1: TAM YEŞİL** — imaj 3.63GB ≤8GB, build+run+offline (model-bake) uçtan uca doğrulandı, §5.4 temiz, **540s süre-bütçesi guard** (10dk timeout koruması), pytest 22/22.
**Baseline (3-video, model entegre):** hepsi valid; **tip=suv (YOLO26-cls top1 0.933)**, renk=siyah, **plaka 34TC8532 3/3**, davranışlar doğru-senaryo (v1 sigara / v2 telefon / v3 slalom).
**Yetenek güncel:**
- ✅ GERÇEK MODEL: plaka(0.983) · sigara(0.856) · kemer(0.895) · **araç-tip 7-sınıf(0.933)** · araç tespit/takip
- ✅ TÜRETİLMİŞ: slalom · renk(HSV) · yolcu(kişi-merkezli)
- 🟡 KISMİ: telefonla_konusma(COCO+pose) · etrafa_bakinma · bilgisayar(COCO laptop)
- ❌ EKSİK (domain-uyumlu veri gerek): esneme · su_icme · arkaya_bakma · teknocan
**Teslim hazır:** teslim/{FTR_GONDERILECEK.pdf(10sf), kod.zip(193MB), TESLIM_NOTU.md} · D-2 §9 7/7.
**Konverjans:** P1-P7 ele alındı; kalan kazanç kullanıcı-girdisi (Roboflow key) ya da regresyon-riskli (stride) → düşük-frekans moduna geçildi.
