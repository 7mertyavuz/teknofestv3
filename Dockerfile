# =============================================================================
# TEKNOFEST 2026 — teknofestv3 çıkarım imajı (D-2 §6, §8 birebir)
# Hedef: NVIDIA Tesla T4 (Turing/sm_75) · base CUDA 12.1 · imaj ≤8GB · ≤10dk
# RUNTIME İNTERNET YOK → tüm bağımlılık + model ağırlıkları + OCR modelleri build'de.
# BOYUT: kurulum+strip+temizlik TEK RUN'da (layer additivity) → ≤8GB.
# =============================================================================
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip binutils ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data/input /app/data/output /app/weights /app/models /app/.ultralytics

# --- TÜM bağımlılıklar + AGRESİF SLİMLEME tek RUN'da (imaj ≤8GB) ---
#  • torch+torchvision cu121 (T4 sm_75) + requirements
#  • .so sembollerini strip (CUDA/torch kütüphaneleri büyük sembol tablosu taşır → ~GB tasarruf)
#  • duplike opencv-python (ultralytics çeker; headless yeter) + test/pyc/static(.a)/header(.h) sil
COPY requirements.txt .
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 \
 && pip3 install --no-cache-dir -r requirements.txt \
 && pip3 uninstall -y opencv-python 2>/dev/null || true \
 && PYDIR=$(python3 -c "import site;print(site.getsitepackages()[0])") \
 && find "$PYDIR" -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true \
 && find "$PYDIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true \
 && find "$PYDIR" -type d \( -name 'tests' -o -name 'test' \) -prune -exec rm -rf {} + 2>/dev/null || true \
 && find "$PYDIR" -name '*.pyc' -delete 2>/dev/null || true \
 && find "$PYDIR" \( -name '*.a' -o -name '*.h' -o -name '*.hpp' \) -delete 2>/dev/null || true \
 && rm -rf /root/.cache/pip /tmp/*

# Kaynak kod + ağırlıklar + config (seçici COPY — D-2 §8)
COPY weights/ /app/weights/
COPY roadguard/ /app/roadguard/
COPY config/ /app/config/
COPY src/ /app/src/
COPY main.py README.md ./

# ÇEVRİMDIŞI MODEL GÖMME (build'de internet var, runtime YOK):
#  Pipeline'ı bir kez ısıt → YOLO26 ağırlıkları + EasyOCR (craft+latin) + fast-plate-ocr
#  (mobile-vit) ONNX modelleri iner ve image'a gömülür. ultralytics sync kapalı.
RUN yolo settings sync=False 2>/dev/null || true
RUN python3 - <<'PY' && rm -rf /root/.cache/pip /tmp/*
import numpy as np
from roadguard.config import load_config
from roadguard.pipeline.pipeline import Pipeline
p = Pipeline(load_config())
for i in range(2):
    p.process_frame(np.full((720, 1280, 3), 127, np.uint8), i)
p.close()
print("warmup+model-bake OK")
PY

CMD ["python3", "main.py"]
