"""D-2 sözleşme testleri — şema doğrulayıcı, plaka normalizasyonu, etiket setleri.

CI'da koşar (`pytest tests/`). Üretilen results.json'un D-2'ye birebir uyduğunu garanti eder.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.d2_labels import (  # noqa: E402
    CATEGORY_LABELS,
    COLORS,
    PLATE_UNREADABLE,
    VEHICLE_TYPES,
    normalize_color,
    normalize_plate,
    normalize_vehicle_type,
    to_ascii_lower,
    validate_results,
)


def _valid_doc():
    return {
        "video_id": "video.mp4",
        "arac_bilgisi": {"tip": "sedan", "plaka": "34TC8532", "renk": "siyah",
                         "confidence_score": 0.94},
        "tespitler": [
            {"zaman_saniye": 14.5, "kategori": "sofor_eylemi",
             "etiket": "telefonla_konusma", "confidence_score": 0.89},
            {"zaman_saniye": 22.1, "kategori": "nesneler",
             "etiket": "bilgisayar", "confidence_score": 0.95},
            {"zaman_saniye": 45.8, "kategori": "yolcular",
             "etiket": "on_koltuk", "confidence_score": 0.91},
        ],
    }


def test_valid_doc_passes():
    assert validate_results(_valid_doc()) == []


def test_label_sets_are_ascii_lower():
    for s in (*VEHICLE_TYPES, *COLORS):
        assert s == s.lower() and all(ord(c) < 128 for c in s)
    for labels in CATEGORY_LABELS.values():
        for s in labels:
            assert s == s.lower() and all(ord(c) < 128 for c in s)


def test_bad_key_detected():
    d = _valid_doc()
    d["arac_bilgisi"]["guven_skoru"] = 0.5  # yanlış anahtar (D-2 §5.1)
    assert any("beklenmeyen anahtar" in e for e in validate_results(d))


def test_bad_label_detected():
    d = _valid_doc()
    d["tespitler"][0]["etiket"] = "telefon"  # geçersiz etiket
    assert validate_results(d)


def test_turkish_char_in_label_detected():
    d = _valid_doc()
    d["tespitler"][0]["etiket"] = "sigara_içme"  # Türkçe karakter
    assert validate_results(d)


def test_confidence_out_of_range_detected():
    d = _valid_doc()
    d["tespitler"][0]["confidence_score"] = 1.7
    assert validate_results(d)


def test_plate_normalization():
    assert normalize_plate("34 TC 8532") == "34TC8532"
    assert normalize_plate("34tc8532") == "34TC8532"
    assert normalize_plate("06 BBC 123") == "06BBC123"
    # geçersiz il kodu (00, 82+) → tespit edilemedi
    assert normalize_plate("00XX1234") == PLATE_UNREADABLE
    assert normalize_plate("99ZZ9999") == PLATE_UNREADABLE
    assert normalize_plate("") == PLATE_UNREADABLE
    assert normalize_plate(None) == PLATE_UNREADABLE
    assert normalize_plate("ABCDEF") == PLATE_UNREADABLE


def test_vehicle_type_normalization():
    assert normalize_vehicle_type("car") == "sedan"
    assert normalize_vehicle_type("truck") == "kamyon"
    assert normalize_vehicle_type("SUV") == "suv"
    assert normalize_vehicle_type(None) == "sedan"
    assert normalize_vehicle_type("uçak") in VEHICLE_TYPES  # bilinmeyen → fallback


def test_color_normalization():
    assert normalize_color("Kırmızı") == "kirmizi"
    assert normalize_color("BEYAZ") == "beyaz"
    assert normalize_color("pembe") is None  # D-2 dışı renk → None
    assert normalize_color(None) is None


def test_ascii_lower_helper():
    assert to_ascii_lower("ŞOFÖR_EYLEMİ") == "sofor_eylemi"
    assert to_ascii_lower("Kırmızı") == "kirmizi"
