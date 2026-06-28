# =============================================================================
# TEKNOFEST 2026 — teknofestv3 çıkarım imajı (D-2 §6, §8 birebir)
# Hedef: NVIDIA Tesla T4 (Turing/sm_75) · base CUDA 12.1 · imaj ≤8GB · ≤10dk
# RUNTIME İNTERNET YOK → tüm bağımlılık + model ağırlıkları + OCR modelleri build'de.
# =============================================================================
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ultralytics: çevrimdışı (sürüm/analitik ping yok), config kalıcı dizin
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLBACKEND=Agg

# Sistem paketleri (D-2 örnek + OpenCV/onnx için libGL/libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data/input /app/data/output /app/weights /app/models /app/.ultralytics

# 1) torch + torchvision — Tesla T4 (sm_75) için cu121 (base CUDA 12.1 ile hizalı).
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1

# 2) Diğer bağımlılıklar (PyPI)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 3) Kaynak kod + ağırlıklar + config (seçici COPY — imaj boyutu kontrolü, D-2 §8)
COPY weights/ /app/weights/
COPY roadguard/ /app/roadguard/
COPY config/ /app/config/
COPY src/ /app/src/
COPY main.py README.md ./

# 4) ÇEVRİMDIŞI MODEL GÖMME (build'de internet var, runtime YOK):
#    Pipeline'ı bir kez kara-kare üzerinde ısıt → tüm YOLO26 ağırlıkları yüklenir,
#    EasyOCR (craft+latin) ve fast-plate-ocr (mobile-vit) ONNX modelleri İNER ve
#    image'a gömülür. ultralytics sync kapatılır (runtime ağ çağrısı olmasın).
RUN yolo settings sync=False 2>/dev/null || true
RUN python3 - <<'PY'
import numpy as np
from roadguard.config import load_config
from roadguard.pipeline.pipeline import Pipeline
p = Pipeline(load_config())
# 2 kare: OCR motoru (fastplate+easyocr) plaka yolunda tetiklensin diye plaka-benzeri desen
for i in range(2):
    p.process_frame(np.full((720, 1280, 3), 127, np.uint8), i)
p.close()
print("warmup+model-bake OK")
PY

# D-2 §8: docker run → otomatik başlar
CMD ["python3", "main.py"]
