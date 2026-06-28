## 1. Proje Özeti

**RoadGuard**, TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" yarışmasının FTR aşaması için geliştirdiğimiz çok aşamalı (kaskad) çıkarım hattıdır. Yol kenarı kameradan gelen videoyu tek geçişte D-2 uyumlu `results.json` dosyasına dönüştürür. Hedef: FPS, çözünürlük ve ışığın değiştiği saha koşullarında hem araç kimliğini (tip, plaka, renk) hem sürücü ihlallerini (sigara, telefon, kemer, slalom) güvenilir tespit etmek.

Mimariyi fps'ten bağımsız bir `cv2` akışı üzerine kurup üç aşamaya böldük (Şekil 1). **Stage-1** YOLO26l ve ByteTrack ile tespit/takip yapar, ardından sürücüyü konuma göre kilitleyen **DriverLock** girer. **Stage-2a** sürücü durumunu çıkarır: YOLO26-pose geometrisini `custom_smoking` ile OR-füzyonuna sokup 16/8 zaman-oylamasıyla kararı kararlı kılar. **Stage-2b** plakayı `custom_license_plate` (YOLO26s) ile bulur, fast-plate-ocr (ONNX) ya da yedek EasyOCR ile okur, ardından TR-normalizasyon, ağırlıklı oy konsensüsü ve dürüstlük zırhlarından geçirir. Araç tipini YOLO26-cls (7 sınıf), rengi HSV baskın-renk ile belirler, çıktıları ID-merkezli biriktirip şema doğrulayıcıdan geçiririz.

**Ana sonuçlar (held-out, dosya-kanıtlı):**

- **Plaka tespiti** (`custom_license_plate.pt`, YOLO26s): mAP50 **0.983** · mAP50-95 0.706 · P 0.982 · R 0.963 · F1 **0.973**. Hattımızın en güçlü halkası.
- **Üç videoda uçtan uca test** (gerçek 4K@50fps, GT plaka `34TC8532`): plakayı **3/3 exact-match (CER 0.0)** okuduk, araç sınıfını **%100** bildik, davranışları doğru senaryolarda yakaladık, şema **%100 geçerli** çıktı.
- **Dağıtım**: temel imaj `nvidia/cuda:12.1.0-base-ubuntu22.04`, hedef Tesla T4 (sm_75). İmaj **3.63 GB** (<8 GB), çalışma anında internet gerektirmiyor. Modelleri build sırasında imaja gömüp çevrimdışı çalıştığını doğruladık.
- **Performans**: Fiziksel T4 olmadığı için kart üzerindeki gerçek ölçümü henüz alamadık. CPU'da bir klipte 75 kareyi 82 s'de işledik. T4 için ~27 FPS bekliyoruz (ham PyTorch × FP16). Abartmamak için bunu raporda "projeksiyon" diye etiketledik.

---

## 2. Veri Seti Oluşturulması

Özel modellerin her birini lisansı açık **açık kaynak veri setleriyle** eğittik, hepsini tek bir birleşik taksonomiye (`train/unified_taxonomy.py`) eşleyip D-2 ile hizaladık. Toplam özel-eğitim görseli **19.264** (9.123 + ~557 + 3.104 + 6.480). Stage-1 tabanını COCO val2017 (5.000) üzerinde doğruladık.

| # | Veri Seti (Kaynak) | Kullanım / Model | Görsel | Lisans |
|---|---|---|---|---|
| 1 | keremberke `license-plate-object-detection` (HF) | Plaka — `custom_license_plate.pt` (YOLO26s) | 9.123 | CC BY 4.0 |
| 2 | CigDet — Cigarette Detection (Mendeley) | Sigara — `custom_smoking.pt` (YOLO26s) | ~557 | CC BY 4.0 |
| 3 | Seatbelt Detection (Roboflow / HF) | Emniyet kemeri — `custom_seatbelt.pt` (YOLO26s) | 3.104 | CC BY 4.0 |
| 4 | QoDe-5G `raw-car-classification-iHasibi` (HF) | Araç tipi (7 D-2 sınıfı) — YOLO26-cls | 6.480 | Açık erişim (HF) |
| 5 | COCO val2017 (Lin vd.) | Stage-1 araç/kişi taban doğrulaması — `yolo26l` (stok) | 5.000 | CC BY 4.0 |

**Etiketleme ve taksonomi:** Kaynak setler etiketli geldiği için emeğimiz birleştirme ve temizlikteydi. ALIAS-remap uyguladık (`cigarette/smoking/smoke → sigara_icme`, `car/saloon → sedan`; DRIVER 6, VEHICLE 7 sınıf), kapsam dışı etiketleri filtreledik (`Bike/CNG → None`), negatif gürültüyü ayıkladık. Türetilmiş davranışlar (`arkaya_bakma`, `etrafa_bakinma`, `slalom`, kemer yokluğu) ayrı bir detection sınıfı değildir. Bunları pose ve geometriden zaman-oylamasıyla çıkarırız.

**Sınıf dengeleme:** Az örnekli araç-tip sınıflarını yalnızca eğitimde, en büyük sınıfın boyutuna kadar çoğalttık (`vehicle_type_cls.py`). Val'i doğal dağılımında bıraktık, böylece ne sızıntı ne yapay şişme oluşur.

**Veri artırma (gürbüzlük):** Augmentasyonu saha videolarındaki FPS, çözünürlük, ışık ve hava çeşitliliğini (D-2 §7) taklit edecek şekilde kurguladık (`train/train.py`, `vehicle_type_cls.py`):

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

Tüm koşumları `seed=0, deterministic=True` ile yaptık, sonuçlar tekrarlanabilir.

**Train/Val/Test (val ≠ test):** Tespit modellerinde kaynaktaki `train`'i eğitime ayırdık, `valid`+`test`'i birleştirip held-out **val** yaptık (§4 metrikleri buradan gelir). Araç-tip sınıflandırmasında sınıf bazında %80/%20 ayrım yaptık (`val_ratio=0.2, seed=0`), oversample yalnızca eğitime. **Bağımsız TEST** için eğitim ve val ile hiç ortak görsel içermeyen, farklı domainden 3 gerçek **4K@50fps** video kullandık. Bu ayrım optimistik yanlılığı yapısal olarak engeller, sızıntı mümkün değildir. Saha sonuçları §4.2'de.

---

## 3.1 Problemin Analizi

Bu problemi laboratuvardan ayıran şey, girdinin kontrolsüz saha koşullarında üretilmesidir. Ne ışığa ne kameraya müdahale edebiliyoruz, araç hareket hâlinde ve kararı tek kareye değil aynı ID'ye ait kare yığınına dayandırmak zorundayız. Tabloda altı temel zorluğu, çözümünü ve gerekçesini topladık.

| Zorluk | İzlenen Çözüm | Gerekçe |
|---|---|---|
| **Işık / parlama** | HSV augmentasyon + random-erasing; HSV ton kanalı; OCR iki-motor OR füzyonu; zemin-koşulu | Parlama tek kareyi yakar, epizodu değil; OR füzyonu düşen motoru kurtarır. |
| **Hareket bulanıklığı** | fps-bağımsız okuma; ByteTrack; 16/8 oylama; min-track kapısı | Bulanık tek kare güvenilmez; çok kareden oy toplanır, geçici track elenir. |
| **Okluzyon** | ByteTrack yeniden ilişkilendirme; alan-ağırlıklı oy; pose + custom_smoking OR | Örtülü (küçük-alanlı) görünüm az ağırlıklanır; OR füzyonu diğer açıya izin verir. |
| **Küçük / uzak plaka** | custom_license_plate (YOLO26s) + sıkı kırpma → OCR; ağırlıklı oy + dürüstlük zırhları; TR-normalizasyon | Darboğaz çözünürlük; sıkı kırpma piksel/karakter bilgisini maksimize eder (mAP50 0.983 / F1 0.973; uçtan uca 3/3 CER 0.0). |
| **Değişken FPS / çözünürlük** | fps-bağımsız örnekleme; scale/translate/degrees + mosaic; zaman damgası gerçek fps'ten | Mantık fps'e gömülürse 25fps'te bozulur; tasarım epizot zamanlamasını profilden ayırır. |
| **Karanlık kabin** | DriverLock; pose el-kulak/ağız geometrisi + custom_smoking OR; 16/8 oylama | Loş kabinde tek-kare gürültülü; karar zaman penceresine yayılan kanıta bağlandı (custom_smoking F1 0.837, custom_seatbelt F1 0.818). |

Üç imza kararımız da aynı fikre dayanır: tek kareye değil, zamanda biriken kanıta güvenmek. **16/8 ID-merkezli zaman-oylaması** gürültüyü histerezisle bastırır. **Sıkı LP kırpma** karakter başına pikseli en üst düzeye çıkarıp CER 0.0'a katkı verir. **Alan-ağırlıklı sınıf oyu** %100 araç-sınıfı doğruluğunu sağlar (hız için bkz. §4.3).

---

## 3.2 Çözüm Mimarisi

Çalıştırma sözleşmemiz sabittir: `docker run` giriş noktasını (`main.py`) başlatıp `/app/data/input/video.mp4` dosyasını `/app/data/output/results.json` dosyasına dönüştürür. Ortama göre davranış değiştiren yapı yoktur, sistem tek şekilde davranır (D-2 §5.4). Uçtan uca akış Şekil 1'de: ham video tek bir Stage-1 tespit/takip katmanından geçer, ardından sürücü durumu, plaka ve araç özellikleri dalları paralel çalışıp ID-merkezli birikimde buluşur.

Stage-1, 2a ve 2b ayrıntıları §3.3'te, held-out sayıları §4.1 Tablo 1'de. Sağlamlık için her kareyi `try/except` ile izole ederiz, tek bozuk kare bütün koşuyu çökertmez. DriverLock yolcuyu `confirm_frames=3` ile kilitler. **ID-merkezli birikim** kare, güven, alan, oy, plaka ve davranış aralıklarını eşikli epizot ayrımıyla birleştirir. Ardışık iki tespit arası 1.2 s'den fazlaysa ayrı olay sayar. **D-2 çıktısı** ana aracı seçer, `arac_bilgisi` alanını ve zaman damgalı `tespitler` listesini şema doğrulayıcıdan geçirir. Çökmede `fallback_doc` boş ama geçerli bir `results.json` üretir.

İç tip ile D-2 çıktısı arasındaki çeviriyi `src/d2_labels`'te izole ettik. Kapsam dışı katmanları (event-stream, QoD/5G, hız tahmini, dashboard) D-2 yolundan bilerek çıkardık, çünkü kaskadın tek hedefe (şema açısından geçerli bir `results.json`) odaklanmasını istedik.

---

## 3.3 Çözüm Detayları

Temel ilke: tam kare yalnızca Stage-1'e girer, sonrası ROI kırpıkları üzerinde çalışır, akış tümüyle ID-merkezlidir. Bağımlılık sürümlerini sabitleyip build'de imaja gömdük, çalışma anında internet gerekmez. Sürümler: PyTorch+torchvision 2.5.1/0.20.1 (cu121, T4 sm_75), Ultralytics 8.4.66 (YOLO26 tespit/pose/cls + ByteTrack), OpenCV-headless 4.10.0.84, fast-plate-ocr 1.1.0 (ONNX `global-plates-mobile-vit-v2`), ONNXRuntime 1.20.1, EasyOCR 1.7.2, Pydantic 2.13.4 (D-2 `schema.py`) ve NumPy 2.1.3.

**Donanım ayrımı:** Eğitimi RTX 5070 (cu128) üzerinde yaptık, dağıtım T4 (sm_75, CUDA 12.1) üzerinde çalışacak. İkisi farklı CUDA ABI'sine sahip olduğundan `roadguard/device.py` gerçek bir kernel çalıştırıp "no kernel image" türü uyumsuzlukları yakalar. Doğrulayamazsa EasyOCR ve fast-plate-ocr dâhil sessizce CPU veya MPS'e geçer.

**Stage-1:** YOLO26l (stok COCO) ve ByteTrack (`persist=True`) çalışır, eşikler conf 0.35, iou 0.45. İki profil var: `server` (imgsz 960, CUDA), `laptop` (yolo26s, imgsz 640, MPS). Süzgeç fail-closed mantığıyla kişileri, tabelaları ve telefon/sigara adaylarını ilgili aşamalara yönlendirir. IoU 0.80'i geçen tespitleri dedup ile eler, hayalet track'lerden kurtuluruz. Alan-ağırlıklı oyu `conf × area_norm` ile hesaplar, eşitlikte deterministik karar veririz. DriverLock'u her karede konuma göre yeniden seçeriz.

**Stage-2a (pose + ikinci model):** Stok COCO telefon/sigara davranışını tek başına üretemez, bu yüzden landmark kütüphanesi gerektirmeyen iki kanallı bir oylama tasarladık. (1) **Pose** (ölçekten bağımsız): bilek↔kulak mesafesi `0.40×yüz_genişliği` altındaysa telefon, bilek↔ağız mesafesi `0.60×yüz_genişliği` altındaysa sigara. Kulak görünmüyorsa karar vermeyiz. (2) **custom_smoking** çıktısını geometrik sigara kararıyla OR'larız. Stok phone modeli güçlü conf görürse geometrik kararı latch ile bastırırız. (3) **ROI:** sıkı kırpma, Lanczos büyütme ve LAB-L üzerinde CLAHE/gamma ile karanlık kabini aydınlatırız. (4) **16/8 oylama:** ham bayrak son 16 karenin en az 8'inde görülürse durum "kararlı aktif" sayılır. `no_seatbelt`'i kemer yokluğundan türetiriz. FP'ye karşı koruma için varsayılan kapalıdır.

**Stage-2b (plaka):** (a) `custom_license_plate.pt` ile sıkı kırpma (conf 0.30, pad %8), küçük plakaları 2× büyütürüz. `lp_h`'yi ham ölçeriz, çünkü hem ağırlığı hem tetiği o besler. (b) **OCR:** önce fast-plate-ocr (ONNX mobile-vit), başarısız olursa EasyOCR. Ortak hatta parlama ve far ışığını reddeder, küçük/karanlık ROI'de CLAHE ve 2× büyütmeyle ikinci şans veririz. (c) **TR-normalizasyon:** plakayı `^\d{2}[A-Z]{1,3}\d{2,4}$` ile ayrıştırır, pozisyona duyarlı düzeltme uygularız (rakam yerinde O→0/I→1/B→8, harf yerinde 0→O), il kodunu 1–81 aralığında doğrularız. (d) **Konsensüs (`PlateVotePool`):** ikamesiz okumaya 1.0, tek ikameliye 0.45, iki ikameliye 0.20, kesik alt-diziye 0.25 ağırlık veririz. Etkin ağırlık `OCR_güveni × kaynak_kalitesi(lp_h)`. (e) **Dürüstlük zırhları** (videoya özel sabit yok): minimum ağırlık 2.0 ile marj ve oran kapısı; zemin koşulu (kazanan plaka en az bir kez net okunmalı); pozisyon vetosu; boyut kapısı. Boyut kapısında `lp_h<13px` ise okuma oya giremez, `<26px` ise sonuç `plate_too_small` işaretlenir.

**Araç özellikleri:** Tipi YOLO26-cls ile belirleriz (7 D-2 sınıfı: sedan, suv, hatchback, pickup, minibus, panelvan, kamyon; model yoksa stok sezgisel girer). Rengi HSV baskın-renk ile buluruz: doygunluk (S) 45 altındaysa akromatik, kromatikse 12 kovalı hue histogramından 9 D-2 rengine ulaşırız. Net ayırt edilemiyorsa alanı boş bırakır, uydurmayız.

**Ön ve son işleme:** Kare seviyesinde fps'ten bağımsızız (`Preprocessor` pass-through; asıl iyileştirme Stage-2 ROI'sinde). Plaka ROI'sinde opsiyonel süper-çözünürlük ve çok kareli median füzyon devreye girebilir. Çıktıyı, tek doğruluk kaynağımız olan ASCII-güvenli şema doğrulayıcıdan geçiririz. Held-out §4.1'de, dağıtım §4.3'te. Araç-tip top-1 doğruluğumuz **0,933** (822 görsel held-out).

---

## 4. Çözümün Sınanması

Sınamayı üç katmanda yürüttük: her modelin **held-out** nicel başarımı; hattın gerçek 4K@50fps videolarda **uçtan uca** entegrasyonu; dağıtım doğrulaması (Docker/T4, internetsiz). **val ≠ test** olduğundan raporladığımız bütün sayılar modelin hiç görmediği held-out veriden gelir.

**Tablo 1. Held-out test başarımı (val ≠ test).** Özel modelleri `model.val` ile ayrı test bölmesinde, stok algılayıcıyı COCO val2017 (5000) üzerinde değerlendirdik.

| Model | Görev | Kaynak | Görsel | mAP@50 | mAP@50-95 | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| custom_license_plate (YOLO26s) | Plaka | keremberke/HF | 9.123 | **0.983** | 0.706 | 0.982 | 0.963 | **0.973** |
| custom_seatbelt (YOLO26s) | Emniyet kemeri | Roboflow/HF | 3.104 | 0.895 | 0.546 | 0.844 | 0.795 | 0.818 |
| custom_smoking (YOLO26s) | Sigara | CigDet/Mendeley | ~557 | 0.856 | 0.457 | 0.855 | 0.820 | 0.837 |
| yolo26l (stok) | Araç + kişi | COCO val2017 | 5.000 | 0.709 | 0.537 | 0.740 | 0.641 | — |
| Araç-tip (YOLO26-cls)† | 7 D-2 tipi | QoDe-5G | 6.480 | top1 **0,933** | top5 0,999 | — | — | — |

Plaka modeli (mAP@50 0.983 / F1 0.973) en kritik halkadır. Sigara modelinin düşük mAP@50-95'i (0.457) küçük nesne tespitinin zorluğunu yansıtır. Bu yüzden onu tek başına değil pose geometrisiyle OR-füzyonu içinde kullanırız. †Araç-tip bir **sınıflandırma** modelidir, metriği mAP değil **top-1 0,933 / top-5 0,999** (822 görsel held-out, 7 D-2 tipi). Entegrasyonda model üç test aracını da **suv** sınıfladı (GT yalnızca "car" içeriyor, ince-tip GT'si yok). Hiçbir yere uydurma değer girmedik.

**Şekil 2: mAP@50 karşılaştırma grafiği.** Dört modelin mAP@50'sini (custom_license_plate 0.983 / seatbelt 0.895 / smoking 0.856 / yolo26l-COCO 0.709) yatay bar grafiğinde gösterdik. Özel modellerin alana özgü performansı stok COCO'nun üzerinde.

**Uçtan uca entegrasyon.** Kaskadda hataların birikip birikmediğini görmek için hattı, GT'sini bildiğimiz gerçek **4K@50fps** videolarda koşturduk.

**Tablo 2. Entegrasyon held-out sonuçları (3 video, GT plaka 34TC8532).**

| Metrik | Sonuç |
|---|---|
| Plaka tam-eşleşme | **3/3 exact-match** |
| Karakter hata oranı (CER) | **0.0** |
| Araç-sınıfı doğruluğu | **%100** |
| Davranış tespiti | Doğru senaryoda (sigara/telefon/slalom yalnız ilgili videoda) |
| Şema doğrulama | **%100 geçerli** |

CER'in 0.0 çıkmasını ağırlıklı oy konsensüsü, dürüstlük zırhları ve TR-normalizasyona borçluyuz. Kare bazlı OCR gürültüsü birikimle bastırılır. Sistem sigara içilmeyen videoda yanlış pozitif üretmedi. %100 geçerli şema, çıktının doğrudan teslim edilebilir olduğunu gösterir.

**Tablo 3. Dağıtım doğrulaması.**

| Kalem | Hedef / Kısıt | Ölçülen | Durum |
|---|---|---|---|
| Docker imaj boyutu | ≤ 8 GB | **3.63 GB** | Geçti |
| Temel imaj | — | nvidia/cuda:12.1.0-base-ubuntu22.04 | Uygun |
| Hedef GPU | Tesla T4 (sm_75) | Uyumlu | Geçti |
| Çevrimdışı çalışma | İnternet yok | Modeller build'de gömülü, offline doğrulandı | Geçti |
| Uçtan-uca koşum | Geçerli results.json | Doğrulandı | Geçti |
| Hız & süre bütçesi | ≤ 10 dk / video | Full-pipeline Apple M-series MPS **~7,5 FPS** (423 kare / 56 s ölçüldü) → tipik 7–9 s test klibi **~56 s** (bütçenin çok altında); detektör (YOLO26l) RTX 5070'te **~27 FPS** | Geçti (bütçe) |

3.63 GB'lik imaj 8 GB sınırının epey altında. Modeller imaja gömülü olduğu için sistem çevrimdışı da tekrarlanabilir sonuç üretir. **Süre bütçesi** açısından rahatız: test klipleri 7–9 s sürüyor (≤450 kare), MPS'te ölçtüğümüz ~7,5 FPS ile ~56 s'de işleniyor, yani 10 dk bütçenin çok altında. Hız konusunda fazlasını iddia etmiyoruz: detektörü RTX 5070'te ~27 FPS ölçtük, ama tam hattın hızı kare başına OCR ve oylama (CPU) maliyetiyle sınırlı. Bunu T4'te `onnxruntime-gpu`, TensorRT FP16 ve kare atlama ile yükseltmeyi planlıyoruz. T4 FPS'ini fiziksel kart elimize geçince ölçeceğiz. Koda bir de **540 s süre-bütçesi guard'ı** ekledik: işleme bu süreyi aşarsa döngüyü kesip o ana kadarki sonuçla geçerli bir `results.json` yazar, böylece timeout'ta çıktısız kalma riskini **sıfırlarız**. Güvenimiz üç ayağa dayanır: görmediği veride yüksek başarım, gerçek videolarda hatasız entegrasyon, hedef donanımda bütçe içinde çevrimdışı çalışma.

---

## 5. Kaynakça

Modellerimizi aşağıdaki açık kaynaklardan türettiğimiz verilerle eğittik. val ≠ test ilkesini koruyarak veri sızıntısını engelledik (D-2 §7).

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
