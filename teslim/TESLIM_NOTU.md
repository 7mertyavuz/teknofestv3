# TEKNOFEST 2026 — teknofestv3 Teslim Notu

## Paket içeriği
- `kod.zip` — yarışma dizin yapısında çalışan kaynak (Dockerfile + main.py + src/ + roadguard/ + config/ + weights/ + requirements.txt + README.md). Hakem bunu açıp `docker build` ile imajı üretir.
- `FTR_GONDERILECEK.pdf` — Final Tasarım Raporu (şablona uygun, gerçek ölçüm tabloları).
- (Opsiyonel) `imaj.tar` — istenirse: `docker build -t teknofest/teknofestv3:latest . && docker save teknofest/teknofestv3:latest -o imaj.tar` (~3.6 GB). Hakem akışı Dockerfile'dan build ettiği için zorunlu değildir.

## Hakem çalıştırma (D-2 §8)
```bash
docker build -t teknofest/teknofestv3:latest .
docker run --rm --gpus all \
  -v <video.mp4>:/app/data/input/video.mp4 \
  -v <cikti-klasoru>:/app/data/output \
  teknofest/teknofestv3:latest
# → /app/data/output/results.json
```

## D-2 §9 Teslim Öncesi Son Kontrol Listesi (DOĞRULANDI)
| Kontrol | Durum |
|---|---|
| Dockerfile projenin en üst dizininde mi? | **E** |
| Temel imaj `nvidia/cuda:12.1.0-base-ubuntu22.04` mı? | **E** |
| Model GPU (cuda) üzerinde çalışacak şekilde mi? | **E** (device auto → T4'te cuda; cu121 torch) |
| `/app/data/input/video.mp4` okunuyor mu? | **E** |
| Sonuçlar `/app/data/output/results.json`'a yazılıyor mu? | **E** |
| Tüm etiketler ASCII + küçük harf mi? | **E** (şema-validatör + pytest) |
| `docker run` ile kod otomatik başlıyor mu? | **E** (`CMD ["python3","main.py"]`) |

### Ek doğrulamalar
- İmaj boyutu: **3.63 GB ≤ 8 GB** (`docker image inspect`).
- Runtime internetsiz: tüm bağımlılık + YOLO26 ağırlıkları + EasyOCR/fast-plate-ocr modelleri **build aşamasında gömülü** (uçtan uca CPU testi: modeller cache'den, indirme yok).
- §5.4: ortam/hostname/IP/env tespitiyle davranış değiştiren kod **YOK** (tek davranış, sabit yol).
- Bozuk/eksik video: kare-başına + global try/except → her durumda geçerli results.json (çökme yok).
- Şema: `tests/test_d2_schema.py` + `tests/test_robustness.py` (CI'da) — anahtar/etiket/ASCII/plaka-regex/conf birebir.

## Drive'a yükleme (public link)
1. `kod.zip` + `FTR_GONDERILECEK.pdf` (gerekirse `imaj.tar`) bir klasöre koy → Google Drive'a yükle.
2. Klasör → Sağ tık → **Paylaş** → "Bağlantısı olan herkes" → **Görüntüleyen** → Bağlantıyı kopyala.
3. Linki KYS'ye gireceğin alana yapıştır; **gizli mod sekmede aç + linkin erişilebilir olduğunu teyit et**.
4. Son teslim saatinden önce yükle (KYS 17:00 kuralı).
