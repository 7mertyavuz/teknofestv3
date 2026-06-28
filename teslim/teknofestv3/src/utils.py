"""Yardımcılar: D-2 results.json yazımı + güvenli fallback belgesi."""

from __future__ import annotations

import json
import os

from src.d2_labels import PLATE_UNREADABLE, validate_results


def fallback_doc(video_id: str) -> dict:
    """Çıkarım tamamen başarısız olsa bile D-2 şemasına UYAN minimal geçerli belge.

    (Bozuk/eksik video, beklenmeyen hata — D-2 §7: çökme yerine geçerli çıktı.)
    """
    return {
        "video_id": video_id or "video.mp4",
        "arac_bilgisi": {
            "tip": None,
            "plaka": PLATE_UNREADABLE,
            "renk": None,
            "confidence_score": 0.0,
        },
        "tespitler": [],
    }


def write_results(doc: dict, output_path: str) -> list[str]:
    """Belgeyi doğrular, çıktı dizinini garanti eder ve results.json yazar.

    Doğrulama ihlalleri DÖNDÜRÜLÜR (loglama için); yazım her durumda yapılır
    (kısmi-geçerli çıktı, hiç çıktı yokluğuna yeğdir). ensure_ascii=False
    (etiketler zaten ASCII; yalnız 'tespit edilemedi' boşluğu için).
    """
    violations = validate_results(doc)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return violations
