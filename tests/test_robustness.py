"""Dayanıklılık + saf-yardımcı testleri (model yüklemeden hızlı koşar).

P4: bozuk/eksik girdi → çökme yok + geçerli çıktı; epizot/koltuk geometrisi mantığı;
fallback + yazıcı roundtrip. (Tam-pipeline entegrasyon testi tests/integration'da.)
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.d2_labels import validate_results  # noqa: E402
from src.predict import _collapse_episodes, _seat_label  # noqa: E402
from src.utils import fallback_doc, write_results  # noqa: E402


# --- fallback + yazıcı (eksik/bozuk video yolu) --- #

def test_fallback_doc_is_valid():
    assert validate_results(fallback_doc("video.mp4")) == []
    assert validate_results(fallback_doc("")) == []


def test_write_results_roundtrip(tmp_path):
    doc = fallback_doc("video_1.mp4")
    out = tmp_path / "sub" / "results.json"
    violations = write_results(doc, str(out))
    assert violations == []
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["video_id"] == "video_1.mp4"
    assert reloaded["arac_bilgisi"]["plaka"] == "tespit edilemedi"
    assert reloaded["tespitler"] == []


def test_write_results_ascii_safe_on_disk(tmp_path):
    # ensure_ascii=False ama etiketler ASCII → dosyada Türkçe karakter sızıntısı olmamalı
    doc = {
        "video_id": "v.mp4",
        "arac_bilgisi": {"tip": "sedan", "plaka": "34TC8532", "renk": "kirmizi",
                         "confidence_score": 0.9},
        "tespitler": [{"zaman_saniye": 1.0, "kategori": "sofor_eylemi",
                       "etiket": "sigara_icme", "confidence_score": 0.8}],
    }
    out = tmp_path / "results.json"
    assert write_results(doc, str(out)) == []
    raw = out.read_text(encoding="utf-8")
    # plaka/etiket alanlarında Türkçe karakter olmamalı
    for bad in "çğıöşüÇĞİÖŞÜ":
        assert bad not in raw


# --- epizot çökeltme (zaman damgalı tespit üretimi) --- #

def test_collapse_episodes_empty():
    assert _collapse_episodes([]) == []


def test_collapse_episodes_single_group_peak():
    # tümü 1.2s boşluk içinde → tek epizot, tepe güven
    out = _collapse_episodes([(1.0, 0.5), (1.5, 0.9), (2.0, 0.6)])
    assert len(out) == 1
    assert out[0] == (1.5, 0.9)


def test_collapse_episodes_two_groups():
    # 5s boşluk → iki ayrı epizot
    out = _collapse_episodes([(1.0, 0.6), (1.4, 0.5), (8.0, 0.8), (8.3, 0.7)])
    assert len(out) == 2
    assert out[0][0] == 1.0 and out[1] == (8.0, 0.8)


# --- koltuk geometrisi --- #

class _BB:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_seat_label_front():
    veh = _BB(0, 0, 100, 100)
    # üst yarı (ry<0.55) → on_koltuk
    assert _seat_label({"bbox": [40, 10, 60, 30]}, veh) == "on_koltuk"


def test_seat_label_back_left_right():
    veh = _BB(0, 0, 100, 100)
    assert _seat_label({"bbox": [10, 70, 30, 90]}, veh) == "arka_koltuk_1"   # sol-arka
    assert _seat_label({"bbox": [70, 70, 90, 90]}, veh) == "arka_koltuk_2"   # sağ-arka


def test_seat_label_missing_bbox():
    assert _seat_label({}, _BB(0, 0, 10, 10)) is None
    assert _seat_label({"bbox": [1, 2, 3, 4]}, None) is None


# --- validatör kenar durumları --- #

def test_validator_missing_video_id():
    d = {"arac_bilgisi": {"tip": "sedan", "plaka": "34TC8532", "renk": "siyah",
                          "confidence_score": 0.5}, "tespitler": []}
    assert any("video_id" in e for e in validate_results(d))


def test_validator_tespitler_not_list():
    d = fallback_doc("v.mp4")
    d["tespitler"] = {"x": 1}
    assert any("tespitler" in e for e in validate_results(d))


def test_validator_bool_confidence_rejected():
    d = fallback_doc("v.mp4")
    d["arac_bilgisi"]["confidence_score"] = True  # bool, float değil
    assert validate_results(d)
