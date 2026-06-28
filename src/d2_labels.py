"""D-2 çıktı sözleşmesi — TEK doğruluk kaynağı.

TEKNOFEST 2026 "5G & YZ ile Akıllı Yol Güvenliği" FTR aşaması teslim dokümanı (D-2)
results.json şemasını, geçerli etiket setlerini, plaka regex'ini ve doğrulayıcıyı
burada tanımlar. Hiçbir etiket/anahtar koda dağınık gömülmez; üretim ve test bu
modülü kullanır → hatasız eşleşme garanti.

KURALLAR (D-2 §2, §3, §5):
  • Tüm kategori/etiket adları ASCII-safe + küçük harf (Türkçe karakter YOK).
  • JSON anahtarları birebir: video_id / arac_bilgisi / tip / plaka / renk /
    confidence_score / tespitler / zaman_saniye / kategori / etiket.
  • confidence_score ∈ [0.0, 1.0] float.
  • Plaka: TR regex'e uy ya da "tespit edilemedi".
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Geçerli değer setleri (D-2 Tablo 1 & Tablo 2)
# --------------------------------------------------------------------------- #

VEHICLE_TYPES: tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "minibus",
    "panelvan",
    "kamyon",
)

COLORS: tuple[str, ...] = (
    "beyaz",
    "siyah",
    "gri",
    "kirmizi",
    "mavi",
    "sari",
    "yesil",
    "turuncu",
    "kahverengi",
)

# kategori → geçerli etiketler
SOFOR_EYLEMI: tuple[str, ...] = (
    "arkaya_bakma",
    "esneme",
    "sigara_icme",
    "su_icme",
    "telefonla_konusma",
    "slalom",
    "etrafa_bakinma",
    "emniyet_kemeri_ihlali",
)
NESNELER: tuple[str, ...] = ("teknocan", "bilgisayar")
YOLCULAR: tuple[str, ...] = ("arka_koltuk_1", "arka_koltuk_2", "on_koltuk")

CATEGORY_LABELS: dict[str, tuple[str, ...]] = {
    "sofor_eylemi": SOFOR_EYLEMI,
    "nesneler": NESNELER,
    "yolcular": YOLCULAR,
}

PLATE_UNREADABLE = "tespit edilemedi"

# D-2 resmi plaka regex'i (§2 Tablo 1) — il kodu 01-81 + harf/hane kombinasyonları.
# Boşluk toleranslı; normalize edilmiş çıktıyı (boşluksuz) da kabul eder.
PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})"
    r"|(\s?[a-zA-Z]{2}\s?)(\d{3,4})"
    r"|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

# --------------------------------------------------------------------------- #
# roadguard iç bayrakları → D-2 (kategori, etiket) eşlemesi
# --------------------------------------------------------------------------- #
# DriverState bayrakları (phone/smoking/no_seatbelt/fatigue) ve SpeedState.swerving.
# NOT: 'fatigue' D-2'nin 8 sofor_eylemi etiketinde DOĞRUDAN yoktur; en yakını 'esneme'
# ama yorgunluk≠esneme → adanmış esneme modeli gelene dek D-2'ye YAZILMAZ (None).
DRIVER_FLAG_TO_D2: dict[str, tuple[str, str] | None] = {
    "smoking": ("sofor_eylemi", "sigara_icme"),
    "phone": ("sofor_eylemi", "telefonla_konusma"),
    "no_seatbelt": ("sofor_eylemi", "emniyet_kemeri_ihlali"),
    "swerving": ("sofor_eylemi", "slalom"),
    "fatigue": None,  # D-2 karşılığı yok (esneme modeli eğitilince eklenecek)
}

# Nesne dedektörü sınıf adı → D-2 nesneler etiketi (COCO 'laptop' → bilgisayar).
OBJECT_CLASS_TO_D2: dict[str, str] = {
    "laptop": "bilgisayar",
    "computer": "bilgisayar",
    "bilgisayar": "bilgisayar",
    "teknocan": "teknocan",
}

# roadguard araç sınıfı (stok COCO/yolo26) → D-2 tip HEURİSTİK FALLBACK eşlemesi.
# Adanmış 7-sınıf araç-tip modeli (weights/vehicle_type.pt) varsa O ezer; bu yalnız
# model yokken kaba tahmindir (test videoları binek araç → sedan ana yol).
VEHICLE_CLASS_TO_D2: dict[str, str] = {
    "car": "sedan",
    "truck": "kamyon",
    "bus": "minibus",
    "minibus": "minibus",
    "van": "panelvan",
    "pickup": "pickup",
    "suv": "suv",
    "hatchback": "hatchback",
    "sedan": "sedan",
    "panelvan": "panelvan",
    "kamyon": "kamyon",
}


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #


def to_ascii_lower(text: str) -> str:
    """Türkçe karakterleri ASCII'ye indirger ve küçük harfe çevirir (etiket güvenliği)."""
    tr_map = str.maketrans(
        {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
         "Ç": "c", "Ğ": "g", "İ": "i", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u"}
    )
    text = text.translate(tr_map)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower().strip()


def clamp_conf(value) -> float:
    """confidence_score'u [0.0, 1.0] float aralığına sabitler."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def round_time(seconds) -> float:
    """zaman_saniye'yi 0.1 sn çözünürlüğe yuvarlar (negatif → 0)."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, s), 1)


def normalize_plate(raw: str | None) -> str:
    """Ham plaka okumasını D-2 biçimine getirir: boşluksuz, BÜYÜK harf, regex-doğrulanmış.

    Regex'i geçemeyen / boş / None okuma → "tespit edilemedi" (D-2 §5.3).
    """
    if not raw:
        return PLATE_UNREADABLE
    # Türkçe harf → ASCII (plakada normalde yok ama OCR güvenliği), boşlukları sil.
    cleaned = to_ascii_lower(str(raw)).upper()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Z]", "", cleaned)
    if PLATE_REGEX.match(cleaned):
        return cleaned
    return PLATE_UNREADABLE


def normalize_vehicle_type(value: str | None) -> str:
    """Araç tipini D-2 geçerli setine eşler; eşlenemezse 'sedan' (en olası binek)."""
    if not value:
        return "sedan"
    key = to_ascii_lower(str(value))
    if key in VEHICLE_TYPES:
        return key
    return VEHICLE_CLASS_TO_D2.get(key, "sedan")


def normalize_color(value: str | None) -> str | None:
    """Renk değerini D-2 geçerli setine sabitler; geçersizse None (renk atlanır)."""
    if not value:
        return None
    key = to_ascii_lower(str(value))
    return key if key in COLORS else None


# --------------------------------------------------------------------------- #
# Doğrulayıcı (CI + e2e harness kullanır) — D-2 §5 Altın Kurallar
# --------------------------------------------------------------------------- #


def validate_results(doc: dict) -> list[str]:
    """results.json belgesini D-2 şemasına karşı doğrular; ihlal listesi döner (boş=geçerli)."""
    errs: list[str] = []

    if not isinstance(doc, dict):
        return ["kök nesne bir JSON object değil"]

    # video_id
    if "video_id" not in doc or not isinstance(doc["video_id"], str) or not doc["video_id"]:
        errs.append("video_id eksik/boş veya string değil")

    # arac_bilgisi
    av = doc.get("arac_bilgisi")
    if not isinstance(av, dict):
        errs.append("arac_bilgisi eksik veya object değil")
    else:
        allowed_keys = {"tip", "plaka", "renk", "confidence_score"}
        for k in av:
            if k not in allowed_keys:
                errs.append(f"arac_bilgisi içinde beklenmeyen anahtar: {k}")
        tip = av.get("tip")
        if tip is not None and tip not in VEHICLE_TYPES:
            errs.append(f"arac_bilgisi.tip geçersiz: {tip!r} (geçerli: {VEHICLE_TYPES})")
        renk = av.get("renk")
        if renk is not None and renk not in COLORS:
            errs.append(f"arac_bilgisi.renk geçersiz: {renk!r}")
        plaka = av.get("plaka")
        if plaka is not None and plaka != PLATE_UNREADABLE:
            if not isinstance(plaka, str) or not PLATE_REGEX.match(plaka):
                errs.append(f"arac_bilgisi.plaka regex/'{PLATE_UNREADABLE}' kuralını ihlal: {plaka!r}")
        cs = av.get("confidence_score")
        if not isinstance(cs, (int, float)) or isinstance(cs, bool) or not (0.0 <= float(cs) <= 1.0):
            errs.append(f"arac_bilgisi.confidence_score [0,1] float değil: {cs!r}")

    # tespitler
    tesp = doc.get("tespitler")
    if not isinstance(tesp, list):
        errs.append("tespitler eksik veya list değil")
    else:
        for i, d in enumerate(tesp):
            if not isinstance(d, dict):
                errs.append(f"tespitler[{i}] object değil")
                continue
            allowed = {"zaman_saniye", "kategori", "etiket", "confidence_score"}
            for k in d:
                if k not in allowed:
                    errs.append(f"tespitler[{i}] beklenmeyen anahtar: {k}")
            zt = d.get("zaman_saniye")
            if not isinstance(zt, (int, float)) or isinstance(zt, bool) or zt < 0:
                errs.append(f"tespitler[{i}].zaman_saniye >=0 sayı değil: {zt!r}")
            kat = d.get("kategori")
            et = d.get("etiket")
            if kat not in CATEGORY_LABELS:
                errs.append(f"tespitler[{i}].kategori geçersiz: {kat!r}")
            elif et not in CATEGORY_LABELS[kat]:
                errs.append(f"tespitler[{i}].etiket '{et}' kategori '{kat}' için geçersiz")
            cs = d.get("confidence_score")
            if not isinstance(cs, (int, float)) or isinstance(cs, bool) or not (0.0 <= float(cs) <= 1.0):
                errs.append(f"tespitler[{i}].confidence_score [0,1] float değil: {cs!r}")

    # ASCII + küçük harf güvenliği (etiket/kategori/tip/renk)
    def _ascii_lower_ok(s: str) -> bool:
        return s == s.lower() and all(ord(c) < 128 for c in s)

    if isinstance(av, dict):
        for fld in ("tip", "renk"):
            val = av.get(fld)
            if isinstance(val, str) and val and not _ascii_lower_ok(val):
                errs.append(f"arac_bilgisi.{fld} ASCII+küçük-harf değil: {val!r}")
    if isinstance(tesp, list):
        for i, d in enumerate(tesp):
            if not isinstance(d, dict):
                continue
            for fld in ("kategori", "etiket"):
                val = d.get(fld)
                if isinstance(val, str) and not _ascii_lower_ok(val):
                    errs.append(f"tespitler[{i}].{fld} ASCII+küçük-harf değil: {val!r}")

    return errs
