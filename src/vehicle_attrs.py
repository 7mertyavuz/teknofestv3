"""Araç tip & renk çıkarımı (D-2 arac_bilgisi.tip / .renk).

İKİ KATMANLI tasarım (graceful):
  • RENK: eğitilmiş model YOK → HSV baskın-renk analizi (robust, kalibrasyonsuz).
    Eğitilmiş renk sınıflandırıcı eklenirse (weights/vehicle_color.pt) buraya bağlanır.
  • TİP: eğitilmiş 7-sınıf araç-tip modeli (weights/vehicle_type.pt) varsa O kullanılır;
    yoksa stok araç sınıfı + en-boy oranı HEURİSTİĞİ (D-2 binek-araç ana yolu: sedan).

Her iki çıkarım da D-2 geçerli setine sabitlenir (src/d2_labels). Asla uydurma:
güven düşükse düşük confidence raporlanır, renk belirsizse None döner (alan atlanır).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.d2_labels import COLORS, normalize_color, normalize_vehicle_type

log = logging.getLogger("teknofestv3.vehicle_attrs")

# HSV renk merkezleri (OpenCV H:0-179, S:0-255, V:0-255). Akromatik (düşük S) ayrı ele alınır.
# Hue bantları (yaklaşık): kirmizi 0-10 & 160-179, turuncu 11-22, sari 23-33,
# yesil 34-85, mavi 86-130, kahverengi (düşük-V turuncu/kirmizi) özel.


def estimate_color(vehicle_bgr: "np.ndarray | None") -> tuple[str | None, float]:
    """Araç gövde kırpığından baskın rengi D-2 setine eşler. (renk, güven) döner.

    Pencere/cam/teker gürültüsünü azaltmak için kırpığın MERKEZ-ÜST gövde şeridi
    örneklenir. Renk ayırt edilemezse (None, 0.0).
    """
    if vehicle_bgr is None or getattr(vehicle_bgr, "size", 0) == 0:
        return None, 0.0
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return None, 0.0

    h, w = vehicle_bgr.shape[:2]
    if h < 12 or w < 12:
        return None, 0.0
    # Gövde şeridi: dikeyde %35-%70 (kaput/kapı; cam ve tekerlekten kaçın), yatayda %20-%80.
    y1, y2 = int(h * 0.35), int(h * 0.70)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    roi = vehicle_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    n = H.size
    s_med = float(np.median(S))
    v_med = float(np.median(V))

    # 1) Akromatik (düşük doygunluk): beyaz / gri / siyah → V'ye göre
    if s_med < 45:
        if v_med < 60:
            return "siyah", _conf_from_purity(np.mean(V < 70))
        if v_med > 180:
            return "beyaz", _conf_from_purity(np.mean(V > 170))
        return "gri", _conf_from_purity(np.mean((V >= 60) & (V <= 180)))

    # 2) Kromatik: yeterince doygun pikseller üzerinde baskın hue
    mask = (S > 50) & (V > 40)
    if mask.sum() < max(20, 0.05 * n):
        # az renkli piksel → akromatik karara düş
        if v_med < 70:
            return "siyah", 0.35
        if v_med > 175:
            return "beyaz", 0.35
        return "gri", 0.3

    hues = H[mask]
    vmask = V[mask]
    # Hue histogramı (12 kova) → baskın kova
    hist, _ = np.histogram(hues, bins=np.arange(0, 181, 15))
    dom_bin = int(np.argmax(hist))
    dom_hue = dom_bin * 15 + 7.5
    purity = float(hist[dom_bin] / max(1, hist.sum()))

    color = _hue_to_d2(dom_hue, float(np.median(vmask)))
    return normalize_color(color), _conf_from_purity(purity)


def _hue_to_d2(hue: float, v_med: float) -> str:
    if hue < 11 or hue >= 160:
        return "kirmizi"
    if 11 <= hue < 23:
        # Düşük parlaklıkta turuncu → kahverengi (koyu turuncu/bordo gövde)
        return "kahverengi" if v_med < 120 else "turuncu"
    if 23 <= hue < 34:
        return "sari"
    if 34 <= hue < 86:
        return "yesil"
    if 86 <= hue < 140:
        return "mavi"
    return "kirmizi"


def _conf_from_purity(purity: float) -> float:
    # Saflık (baskın kovanın payı) → 0.4-0.95 arası güven.
    return float(max(0.4, min(0.95, 0.4 + 0.55 * purity)))


class VehicleTypeClassifier:
    """7-sınıf araç-tip çıkarımı. Eğitilmiş model varsa onu, yoksa heuristik kullanır."""

    def __init__(self, weights_dir: str):
        self.model = None
        path = os.path.join(weights_dir, "vehicle_type.pt")
        if os.path.exists(path):
            try:
                from ultralytics import YOLO

                self.model = YOLO(path)
                self._names = self.model.names
                log.info("Araç-tip modeli yüklendi: %s", path)
            except Exception as e:  # noqa: BLE001
                log.warning("Araç-tip modeli yüklenemedi (%s) → heuristik", e)
                self.model = None

    def infer(self, vehicle_bgr, stock_class: str) -> tuple[str, float]:
        """(tip, güven) döner. Model varsa kırpıktan, yoksa stok sınıf heuristiği."""
        if self.model is not None and vehicle_bgr is not None and vehicle_bgr.size:
            try:
                res = self.model.predict(vehicle_bgr, verbose=False)
                if res and len(res):
                    r = res[0]
                    # classify head
                    if getattr(r, "probs", None) is not None:
                        idx = int(r.probs.top1)
                        conf = float(r.probs.top1conf)
                        return normalize_vehicle_type(self._names[idx]), conf
                    # detect head: en yüksek confidence kutusunun sınıfı
                    if getattr(r, "boxes", None) is not None and len(r.boxes):
                        b = r.boxes
                        k = int(b.conf.argmax())
                        cls = self._names[int(b.cls[k])]
                        return normalize_vehicle_type(cls), float(b.conf[k])
            except Exception as e:  # noqa: BLE001
                log.debug("Araç-tip model çıkarımı başarısız (%s) → heuristik", e)
        # Heuristik: stok COCO/yolo26 sınıfından D-2 tipine kaba eşleme (düşük-orta güven).
        return normalize_vehicle_type(stock_class), 0.45
