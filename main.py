"""TEKNOFEST 2026 — teknofestv3 çıkarım giriş noktası (D-2 §8).

docker run ... → otomatik başlar:
  /app/data/input/video.mp4  →  /app/data/output/results.json

KURAL (D-2 §5.4): Ortam (hostname/IP/env/dosya) tespit edip DAVRANIŞ değiştiren
hiçbir yapı YOKTUR. Yollar sabittir; tek davranış vardır.
"""

from __future__ import annotations

import json
import logging
import os
import sys

# src/ ve roadguard/ paketlerine erişim (repo kökü)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

INPUT_PATH = "/app/data/input/video.mp4"
OUTPUT_PATH = "/app/data/output/results.json"
WEIGHTS_PATH = "/app/weights"

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("teknofestv3.main")


def main() -> int:
    # SABİT yollar — tek davranış, ortam/env'e göre dallanma YOK (D-2 §5.4).
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    weights_path = WEIGHTS_PATH
    video_id = os.path.basename(input_path)

    from src.utils import fallback_doc, write_results

    if not os.path.exists(input_path):
        log.error("Girdi videosu bulunamadı -> %s ; boş-geçerli results.json yazılıyor", input_path)
        write_results(fallback_doc(video_id), output_path)
        return 1

    log.info("Yol Güvenliği YZ çıkarımı başladı | girdi=%s | ağırlıklar=%s", input_path, weights_path)
    try:
        from src.predict import run_inference

        doc = run_inference(input_path, weights_path)
    except Exception as e:  # noqa: BLE001 — hiçbir koşulda çıktısız çökme yok (D-2 §7)
        log.exception("Çıkarım başarısız; fallback results.json yazılıyor: %s", e)
        doc = fallback_doc(video_id)

    violations = write_results(doc, output_path)
    if violations:
        log.warning("D-2 şema uyarıları (%d): %s", len(violations), "; ".join(violations[:10]))
    n = len(doc.get("tespitler", []))
    av = doc.get("arac_bilgisi", {})
    log.info(
        "Tamamlandı → %s | tip=%s renk=%s plaka=%s tespit=%d",
        output_path, av.get("tip"), av.get("renk"), av.get("plaka"), n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
