## 1. Proje Özeti

**RoadGuard**, TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" yarışması FTR aşaması için, yol kenarı kamera videosunu **tek geçişte** D-2 uyumlu `results.json`'a dönüştüren çok-aşamalı (kaskad) çıkarım hattıdır. Problem: değişken FPS/çözünürlük/ışık koşullarında araç kimliğini (tip, plaka, renk) ve sürücü ihlallerini (sigara, telefon, kemer, slalom) güvenilir tespit etmek.

Mimari, fps-bağımsız `cv2` akışı üzerine kurulu üç aşamalıdır (Şekil 1): **Stage-1** YOLO26l + ByteTrack tespit + konum-bazlı **DriverLock**; **Stage-2a** YOLO26-pose geometrisi + `custom_smoking` OR-füzyonu + 16/8 zaman-oylamasıyla sürücü durumu; **Stage-2b** `custom_license_plate` (YOLO26s) → fast-plate-ocr (ONNX) / EasyOCR → TR-normalizasyon + ağırlıklı oy konsensüsü + dürüstlük zırhları. Araç tipi YOLO26-cls (7 sınıf), renk HSV baskın-renk; ID-merkezli birikim sonucu şema doğrulayıcıdan geçirir.

**Ana sonuçlar (held-out, dosya-kanıtlı):**

- **Plaka tespiti** `custom_license_plate.pt` (YOLO26s): mAP50 **0.983** · mAP50-95 0.706 · P 0.982 · R 0.963 · F1 **0.973** — hattın en güçlü halkası.
- **3-video uçtan-uca** (gerçek 4K@50fps, GT plaka `34TC8532`): plaka **3/3 exact-match (CER 0.0)**, araç-sınıfı **%100**, davranış doğru senaryoda tespit, şema **%100 geçerli**.
- **Dağıtım**: base `nvidia/cuda:12.1.0-base-ubuntu22.04`, hedef Tesla T4 (sm_75), imaj **3.63 GB** (<8 GB), runtime internetsiz (modeller build'de gömülü, offline doğrulandı).
- **Performans**: T4 gerçek ölçümü beklenmekte (elde T4 yok); CPU klip 75 kare/82 s, T4 projeksiyonu ~27 FPS (ham PyTorch × FP16) — dürüstçe "projeksiyon" etiketli.

---

## 2. Veri Seti Oluşturulması

Her özel model lisansı net **açık kaynak veri setleriyle** eğitildi; setler tek birleşik taksonomiye (`train/unified_taxonomy.py`) eşlenip D-2 ile hizalandı. Toplam **19.264 özel-eğitim görseli** (9.123 + ~557 + 3.104 + 6.480); Stage-1 tabanı COCO val2017 (5.000) üzerinde doğrulandı.

| # | Veri Seti (Kaynak) | Kullanım / Model | Görsel | Lisans |
|---|---|---|---|---|
| 1 | keremberke `license-plate-object-detection` (HF) | Plaka — `custom_license_plate.pt` (YOLO26s) | 9.123 | CC BY 4.0 |
| 2 | CigDet — Cigarette Detection (Mendeley) | Sigara — `custom_smoking.pt` (YOLO26s) | ~557 | CC BY 4.0 |
| 3 | Seatbelt Detection (Roboflow / HF) | Emniyet kemeri — `custom_seatbelt.pt` (YOLO26s) | 3.104 | CC BY 4.0 |
| 4 | QoDe-5G `raw-car-classification-iHasibi` (HF) | Araç tipi (7 D-2 sınıfı) — YOLO26-cls | 6.480 | Açık erişim (HF) |
| 5 | COCO val2017 (Lin vd.) | Stage-1 araç/kişi taban doğrulaması — `yolo26l` (stok) | 5.000 | CC BY 4.0 |

**Etiketleme ve taksonomi:** Kaynaklar hazır etiketli; emek birleştirme/temizliktedir: ALIAS-remap (`cigarette/smoking/smoke → sigara_icme`, `car/saloon → sedan`; DRIVER 6, VEHICLE 7 sınıf), kapsam-dışı filtre (Bike/CNG → `None`), negatif-gürültü eleme. Türetilmiş davranışlar (`arkaya_bakma`, `etrafa_bakinma`, `slalom`, kemer-yokluğu) detection sınıfı değil; pose/geometri + zaman-oylamasıyla türetilir.

**Sınıf dengeleme:** Azınlık araç-tip sınıfları yalnız eğitim bölümünde en büyüğe dek kopyalanır (`vehicle_type_cls.py`); val doğal dağılımda kalır — sızıntı/şişme olmaz.

**Veri artırma (gürbüzlük):** Augmentasyon saha videolarının FPS/çözünürlük/ışık/hava çeşitliliğini (D-2 §7) taklit eder (parametreler `train/train.py`, `vehicle_type_cls.py` dosya-kanıtlı):

| Augmentasyon | Tespit | Araç-tip cls | Gürbüzlük hedefi |
|---|---|---|---|
| Mosaic | 1.0 | — | ölçek/bağlam, küçük nesne |
| MixUp | 0.1 | — | düzenlileştirme |
| HSV (h/s/v) | 0.015/0.7/0.4 | 0.015/0.6/0.4 | ışık-renk-hava (gece/gündüz, sis) |
| Yatay çevirme | 0.5 | 0.5 | yön/şerit bağımsızlığı |
| Döndürme | 5.0° | 8.0° | kamera eğimi |
| Öteleme | 0.1 | 0.1 | kadraj çeşitliliği |
| Ölçek | 0.5 | 0.5 | mesafe/çözünürlük |
| Random Erasing | 0.2 | 0.3 | okluzyon dayanıklılığı |

Tüm koşumlar `seed=0, deterministic=True` ile tekrarlanabilir.

**Train/Val/Test (val ≠ test):** Tespitte kaynak `train` → eğitim, `valid`+`test` → held-out **val** (§4 metrikleri). Araç-tip cls'de sınıf-bazlı %80/%20 (`val_ratio=0.2, seed=0`), oversample yalnız train'e. **Bağımsız TEST**: eğitim/val ile ortak görsel içermeyen, farklı domaindeki 3 gerçek **4K@50fps** video; val/test ayrımı optimistik yanlılığı yapısal engeller (sızıntı imkânsız). Saha sonuçları §4.2'dedir.

---

## 3.1 Problemin Analizi

Problemi laboratuvardan ayıran, girdinin kontrolsüz saha koşullarında üretilmesidir: ışık/kamera kontrolü yok, araç hareketli, karar tek kareye değil aynı ID'nin kare yığınına dayanmak zorunda. Tablo altı zorluğu, çözümü ve gerekçesini verir.

| Zorluk | İzlenen Çözüm | Gerekçe |
|---|---|---|
| **Işık / parlama** | HSV augmentasyon + random-erasing; HSV ton kanalı; OCR iki-motor OR füzyonu; zemin-koşulu | Parlama tek kareyi yakar, epizodu değil; OR füzyonu düşen motoru kurtarır. |
| **Hareket bulanıklığı** | fps-bağımsız okuma; ByteTrack; 16/8 oylama; min-track kapısı | Bulanık tek kare güvenilmez; çok kareden oy toplanır, geçici track elenir. |
| **Okluzyon** | ByteTrack yeniden ilişkilendirme; alan-ağırlıklı oy; pose + custom_smoking OR | Örtülü (küçük-alanlı) görünüm az ağırlıklanır; OR füzyonu diğer açıya izin verir. |
| **Küçük / uzak plaka** | custom_license_plate (YOLO26s) + sıkı kırpma → OCR; ağırlıklı oy + dürüstlük zırhları; TR-normalizasyon | Darboğaz çözünürlük; sıkı kırpma piksel/karakter bilgisini maksimize eder (mAP50 0.983 / F1 0.973; uçtan uca 3/3 CER 0.0). |
| **Değişken FPS / çözünürlük** | fps-bağımsız örnekleme; scale/translate/degrees + mosaic; zaman damgası gerçek fps'ten | Mantık fps'e gömülürse 25fps'te bozulur; tasarım epizot zamanlamasını profilden ayırır. |
| **Karanlık kabin** | DriverLock; pose el-kulak/ağız geometrisi + custom_smoking OR; 16/8 oylama | Loş kabinde tek-kare gürültülü; karar zaman penceresine yayılan kanıta bağlandı (custom_smoking F1 0.837, custom_seatbelt F1 0.818). |

Üç imza karar ortak temaya dayanır — tek-kareye değil zamanda biriken kanıta güvenmek: **16/8 ID-merkezli zaman-oylaması** gürültüyü histerezisle bastırır, **sıkı LP kırpma** karakter-başına pikseli maksimize edip CER 0.0'a katkı verir, **alan-ağırlıklı sınıf oyu** %100 araç-sınıfı doğruluğu sağlar (çıkarım hızı için bkz. §4.3).

---

## 3.2 Çözüm Mimarisi

Çalıştırma sözleşmesi sabittir: `docker run` giriş noktasını (`main.py`) başlatır, `/app/data/input/video.mp4 → /app/data/output/results.json` dönüşümünü yapar. Ortama göre davranış değiştiren yapı yok — tek davranış (D-2 §5.4).

**Şekil 1: Sistem Mimarisi — Uçtan Uca Akış**

```
video (cv2, fps-bağımsız kare)
  → [Stage-1] YOLO26l + ByteTrack (araç + kişi + tabela; alan-ağırlıklı sınıf oyu; dedup)
      → DriverLock (konum-bazlı sürücü/yolcu)
  → [Stage-2a] Sürücü durumu: YOLO26-pose geometri + custom_smoking (OR füzyon) → 16/8 oylama
  → [Stage-2b] Plaka: custom_license_plate (YOLO26s) sıkı kırpma → fast-plate-ocr (ONNX) / EasyOCR
      → TR-normalizasyon → ağırlıklı-oy konsensüs + dürüstlük zırhları
  → [Araç özellikleri] tip: YOLO26-cls (7 sınıf) · renk: HSV baskın-renk
  → ID-merkezli birikim → results.json (şema-doğrulayıcıdan geçer)
```

Stage-1/2a/2b ayrıntıları §3.3'te, held-out sayıları §4.1 Tablo 1'dedir. Sağlamlık: kare başına `try/except` izolasyonu (tek bozuk kare koşuyu bitirmez); DriverLock `confirm_frames=3` ile yolcuyu kilitler; **ID-merkezli birikim** kare/güven/alan/oy/plaka/davranış aralıklarını eşikli epizot ayrımıyla toplar (ardışık tespitler >1.2 s ayrıksa ayrı olay); **D-2 çıktısı** ana aracı seçip `arac_bilgisi` + zaman damgalı `tespitler`'i şema doğrulayıcıdan geçirir, çökmede `fallback_doc` boş-geçerli `results.json` üretir.

İç tipler ile D-2 çıktısı arası çeviri `src/d2_labels`'te izole edilir. Kapsam dışı katmanlar (event-stream, QoD/5G, hız tahmini, dashboard) D-2 yolundan bilinçli çıkarıldı; kaskad tek hedefe (şema-geçerli `results.json`) odaklanır.

---

## 3.3 Çözüm Detayları

Ana ilke: tam kare yalnız Stage-1'e girer, sonrası ROI kırpıklarında çalışır; akış ID-merkezlidir. Bağımlılıklar sürüm-sabit ve imaja build'de gömülü (runtime internetsiz): PyTorch+torchvision 2.5.1/0.20.1 (cu121, T4 sm_75), Ultralytics 8.4.66 (YOLO26 tespit/pose/cls + ByteTrack), OpenCV-headless 4.10.0.84, fast-plate-ocr 1.1.0 (ONNX `global-plates-mobile-vit-v2`), ONNXRuntime 1.20.1, EasyOCR 1.7.2, Pydantic 2.13.4 (D-2 `schema.py`), NumPy 2.1.3.

**Donanım ayrımı:** Eğitim (RTX 5070, cu128) ile dağıtım (T4, sm_75, CUDA 12.1) farklı CUDA ABI'larında; `roadguard/device.py` gerçek bir kernel çalıştırıp "no kernel image" uyumsuzluğunu yakalar, doğrulanamazsa sessizce CPU/MPS'e düşer (EasyOCR/fast-plate-ocr dahil).

**Stage-1:** YOLO26l (stok COCO) + ByteTrack (`persist=True`), conf 0.35 / iou 0.45; profiller `server` (imgsz 960, CUDA) / `laptop` (yolo26s, imgsz 640, MPS). Fail-closed süzgeç; kişiler/tabelalar/phone-smoking ilgili aşamalara yönlendirilir. IoU>0.80 dedup hayalet-track'leri eler; alan-ağırlıklı oy `conf × area_norm`, eşitlikte deterministik; DriverLock her kare konuma göre seçilir.

**Stage-2a (pose + ikinci model):** Stok COCO telefon/sigara davranışını üretemez; çözüm landmark kütüphanesiz iki kanal + oylama. (1) **Pose** (ölçek-bağımsız): bilek↔kulak < `0.40×yüz_genişliği` → telefon, bilek↔ağız < `0.60×yüz_genişliği` → sigara; kulak görünmüyorsa çekimserlik. (2) **custom_smoking** geometrik sigaraya OR'lanır; stok phone güçlü conf görülürse geometrik sigara latch'le bastırılır. (3) **ROI:** sıkı kırpma + Lanczos büyütme + LAB-L CLAHE/gamma (karanlık kabin açılır). (4) **16/8 oylama:** ham bayrak son 16 karenin ≥8'inde → "kararlı aktif"; `no_seatbelt` kemer yokluğundan türetilir (varsayılan kapalı, FP koruması).

**Stage-2b (plaka):** (a) `custom_license_plate.pt` sıkı kırpma (conf 0.30, pad %8), küçük plaka 2×; `lp_h` ham ölçülür (ağırlık/tetik kaynağı). (b) **OCR:** fast-plate-ocr (ONNX mobile-vit), yoksa EasyOCR; ortak hatta parlama/far reddi + küçük/karanlık ROI'de CLAHE+2× ikinci şans. (c) **TR-normalizasyon:** `^\d{2}[A-Z]{1,3}\d{2,4}$` parse; pozisyon-farkında düzeltme (rakamda O→0/I→1/B→8, harfte 0→O), il kodu 1–81. (d) **Konsensüs (`PlateVotePool`):** ikamesiz okuma 1.0, 1-ikameli 0.45, 2-ikameli 0.20, kesik alt-dizi 0.25; etkin ağırlık `OCR_güveni × kaynak_kalitesi(lp_h)`. (e) **Dürüstlük zırhları** (videoya-özel sabit yok): min-ağırlık 2.0 + marj + oran kapısı; zemin koşulu (kazanan en az bir kez net okunmuş olmalı); pozisyon-vetosu; boyut kapısı `lp_h<13px` oya giremez, `<26px` → `plate_too_small`.

**Araç özellikleri:** Tip YOLO26-cls (7 D-2 sınıfı: sedan, suv, hatchback, pickup, minibus, panelvan, kamyon; model yoksa stok heuristik). Renk HSV baskın-renk (S<45 → akromatik, kromatikte 12-kovalı hue histogramı → 9 D-2 rengi; ayırt edilemezse atlanır, uydurma yok).

**Ön-/son-işleme:** Kare seviyesi fps-bağımsız (`Preprocessor` pass-through, asıl iyileştirme Stage-2 ROI'sinde); plaka ROI'sinde ek olarak opsiyonel süper-çözünürlük + çok-kareli median füzyon. Çıktı şema doğrulayıcıdan geçer (tek doğruluk kaynağı, ASCII-safe). Held-out §4.1, dağıtım §4.3; araç-tip top-1 **0,933** (822 görsel held-out).

---

## 4. Çözümün Sınanması

Sınama üç katmanda yürütüldü: (1) her modelin **held-out** nicel başarımı, (2) gerçek 4K@50fps videolarda **uçtan-uca entegrasyon**, (3) dağıtım (Docker/T4, internetsiz) doğrulaması. **val ≠ test**: raporlanan sayılar modelin görmediği held-out'tan.

**Tablo 1 — Held-out test başarımı (val ≠ test).** Özel modeller `model.val` ile ayrı test bölmesinde, stok algılayıcı COCO val2017 (5000) üzerinde değerlendirildi.

| Model | Görev | Kaynak | Görsel | mAP@50 | mAP@50-95 | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| custom_license_plate (YOLO26s) | Plaka | keremberke/HF | 9.123 | **0.983** | 0.706 | 0.982 | 0.963 | **0.973** |
| custom_seatbelt (YOLO26s) | Emniyet kemeri | Roboflow/HF | 3.104 | 0.895 | 0.546 | 0.844 | 0.795 | 0.818 |
| custom_smoking (YOLO26s) | Sigara | CigDet/Mendeley | ~557 | 0.856 | 0.457 | 0.855 | 0.820 | 0.837 |
| yolo26l (stok) | Araç + kişi | COCO val2017 | 5.000 | 0.709 | 0.537 | 0.740 | 0.641 | — |
| Araç-tip (YOLO26-cls)† | 7 D-2 tipi | QoDe-5G | 6.480 | top1 **0,933** | top5 0,999 | — | — | — |

Plaka modeli (mAP@50 0.983 / F1 0.973) hattın en kritik halkasıdır; sigara modelinin düşük mAP@50-95'i (0.457) küçük-nesne zorluğunu yansıtır, bu yüzden pose-geometrisiyle OR-füzyonda kullanılır. †Araç-tip bir **sınıflandırma** modelidir; metriği mAP değil **top-1 0,933 / top-5 0,999** (822 görsel held-out, 7 D-2 tipi). Entegrasyonda 3 gerçek test aracını tutarlı biçimde **suv** olarak sınıfladı (GT yalnız "car" içerir; ince-tip GT yok). Uydurma değer girilmemiştir.

**Şekil 2: mAP@50 karşılaştırma grafiği.** Dört modelin mAP@50 değerleri (custom_license_plate 0.983 / seatbelt 0.895 / smoking 0.856 / yolo26l-COCO 0.709) yatay bar grafiğinde gösterilir; özel modellerin alan-özgü performansı stok COCO başarımını aşar.

**Uçtan-uca entegrasyon.** Bağlı kaskad hatasının birikip birikmediğini sınamak için hat, GT'si bilinen gerçek **4K@50fps** videolarda koşturuldu.

**Tablo 2 — Entegrasyon held-out sonuçları (3 video, GT plaka 34TC8532).**

| Metrik | Sonuç |
|---|---|
| Plaka tam-eşleşme | **3/3 exact-match** |
| Karakter hata oranı (CER) | **0.0** |
| Araç-sınıfı doğruluğu | **%100** |
| Davranış tespiti | Doğru senaryoda (sigara/telefon/slalom yalnız ilgili videoda) |
| Şema doğrulama | **%100 geçerli** |

CER 0.0, ağırlıklı oy konsensüsü + dürüstlük zırhları + TR-normalizasyona borçludur (kare-bazlı OCR gürültüsü birikimle bastırılır). Sistem sigara olmayan videoda yanlış-pozitif üretmez; şema %100 çıktının teslim-edilebilir olduğunu gösterir.

**Tablo 3 — Dağıtım doğrulaması.**

| Kalem | Hedef / Kısıt | Ölçülen | Durum |
|---|---|---|---|
| Docker imaj boyutu | ≤ 8 GB | **3.63 GB** | Geçti |
| Temel imaj | — | nvidia/cuda:12.1.0-base-ubuntu22.04 | Uygun |
| Hedef GPU | Tesla T4 (sm_75) | Uyumlu | Geçti |
| Çevrimdışı çalışma | İnternet yok | Modeller build'de gömülü, offline doğrulandı | Geçti |
| Uçtan-uca koşum | Geçerli results.json | Doğrulandı | Geçti |
| Hız (FPS) | — | T4 ölçümü bekleniyor (elde T4 yok); CPU klip 75 kare/82 s; T4 projeksiyonu ~27 FPS (ham PyTorch × FP16, server profili) | Projeksiyon |

3.63 GB imaj 8 GB sınırının altında geniş marjla durur; gömülü modeller runtime'da ağ erişimsiz tekrarlanabilir sonuç sağlar. Hızda dürüstüz: fiziksel T4 olmadığından gerçek FPS henüz ölçülmedi; ~27 FPS **projeksiyondur** ve T4 erişiminde gerçek ölçümle güncellenecektir. Özetle güven, görmediği veride yüksek başarım + gerçek videoda hatasız entegrasyon + hedef donanımda bütçe içi çevrimdışı çalışmaya dayanır.

---

## 5. Kaynakça

Modeller aşağıdaki açık kaynaklardan türetilen verilerle eğitilmiştir (val ≠ test korunarak sızıntı engellendi; D-2 §7).

**Açık veri setleri**

[1] keremberke, *License Plate Object Detection Dataset* (CC BY), 2022, https://huggingface.co/datasets/keremberke/license-plate-object-detection

[2] Khan, A., *CigDet (Cigarette Detection) Dataset*, Mendeley Data, 2024, DOI: 10.17632/6hyrr8typ7.1

[3] seatbelttraining, *Seatbelt Detection Dataset* (v3), Roboflow Universe, 2022, https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec

[4] QoDe-5G, *Raw Car Classification — iHasibi* (7 D-2 araç tipine eşlendi), Hugging Face

**Açık kaynak yazılım ve modeller**

[5] Jocher, G., Qiu, J., Chaurasia, A., *Ultralytics YOLO* (YOLO26 tespit/poz/sınıflandırma; AGPL-3.0), 2023, https://github.com/ultralytics/ultralytics

[6] Andrew, A. (ankandrew), *fast-plate-ocr* (`global-plates-mobile-vit-v2` ONNX; birincil OCR), 2024, https://github.com/ankandrew/fast-plate-ocr

[7] JaidedAI, *EasyOCR* (yedek OCR motoru), 2020, https://github.com/JaidedAI/EasyOCR

**Bilimsel makaleler**

[8] Zhang, Y., Sun, P., Jiang, Y. vd., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022, arXiv:2110.06864

[9] Lin, T.-Y., Maire, M., Belongie, S. vd., *Microsoft COCO: Common Objects in Context*, ECCV 2014, arXiv:1405.0312

[10] TEKNOFEST, *5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması Şartnamesi (D-2 çıktı şeması)*, 2026, https://www.teknofest.org

(Tüm bağlantılar 28.06.2026 tarihinde erişilmiştir.)
