"""Birleşik eğitim taksonomisi — D-2 ile hizalı, AZ sayıda çok-sınıflı model.

"Her şeye ayrı model" YOK: iki YOLO26 modeli eğitilir —
  • DRIVER  → sürücü-eylemi + kabin-nesnesi (tek çok-sınıflı detection)
  • VEHICLE → araç tipi (7-sınıf detection)

Farklı veri setlerinin ham sınıf adları ALIAS ile bu kanonik uzaya eşlenir.
arkaya_bakma / etrafa_bakinma / slalom / yolcu-koltuğu pose/geometri ile türetilir
(detection sınıfı DEĞİL) → bilinçli olarak burada yer almaz.
"""

from __future__ import annotations

# --- DRIVER modeli: kanonik sınıflar (index = eğitim sınıf id'si) ---
DRIVER_CLASSES: list[str] = [
    "sigara_icme",        # cigarette / smoking
    "telefonla_konusma",  # phone / cell phone / calling
    "su_icme",            # drinking / bottle-to-mouth
    "esneme",             # yawn / yawning (yorgunluk göstergesi)
    "emniyet_kemeri",     # seatbelt ŞERİDİ görünür (yokluğu → ihlal türetilir)
    "bilgisayar",         # laptop / computer
]

# --- VEHICLE modeli: D-2 7-sınıf ---
VEHICLE_CLASSES: list[str] = [
    "sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon",
]

# Ham (dataset) sınıf adı → kanonik DRIVER sınıfı. Lowercase eşleşir; bilinmeyen → atlanır.
DRIVER_ALIASES: dict[str, str] = {
    # sigara
    "cigarette": "sigara_icme", "smoking": "sigara_icme", "smoke": "sigara_icme",
    "sigara": "sigara_icme", "cig": "sigara_icme", "driver_smoking": "sigara_icme",
    # telefon
    "phone": "telefonla_konusma", "cell phone": "telefonla_konusma",
    "cellphone": "telefonla_konusma", "mobile": "telefonla_konusma",
    "using_phone": "telefonla_konusma", "talking_phone": "telefonla_konusma",
    "phone_use": "telefonla_konusma", "telefon": "telefonla_konusma",
    # su/içecek
    "drinking": "su_icme", "drink": "su_icme", "bottle": "su_icme",
    "water": "su_icme", "su_icme": "su_icme", "beverage": "su_icme",
    # esneme/yorgunluk
    "yawn": "esneme", "yawning": "esneme", "esneme": "esneme",
    "drowsy": "esneme", "drowsiness": "esneme", "fatigue": "esneme", "tired": "esneme",
    # kemer
    "seatbelt": "emniyet_kemeri", "seat_belt": "emniyet_kemeri",
    "belt": "emniyet_kemeri", "seatbelt_on": "emniyet_kemeri",
    "wearing_seatbelt": "emniyet_kemeri", "emniyet_kemeri": "emniyet_kemeri",
    # bilgisayar
    "laptop": "bilgisayar", "computer": "bilgisayar", "bilgisayar": "bilgisayar",
}

# Ham sınıf adı → kanonik VEHICLE tipi. (büyük/uzun araçlar kamyon; van→panelvan vb.)
VEHICLE_ALIASES: dict[str, str] = {
    "sedan": "sedan", "saloon": "sedan", "car": "sedan",
    "suv": "suv", "jeep": "suv", "crossover": "suv",
    "hatchback": "hatchback", "hatch": "hatchback",
    "pickup": "pickup", "pick-up": "pickup", "pick_up": "pickup", "truck-pickup": "pickup",
    "minibus": "minibus", "van-minibus": "minibus", "midibus": "minibus", "minivan": "minibus",
    "panelvan": "panelvan", "van": "panelvan", "panel van": "panelvan", "cargo_van": "panelvan",
    "kamyon": "kamyon", "truck": "kamyon", "lorry": "kamyon", "camion": "kamyon", "bus": "kamyon",
}

GROUPS = {
    "driver": {"classes": DRIVER_CLASSES, "aliases": DRIVER_ALIASES},
    "vehicle": {"classes": VEHICLE_CLASSES, "aliases": VEHICLE_ALIASES},
}


def canonical_index(group: str, raw_name: str) -> int | None:
    """Ham sınıf adını grup-içi kanonik indekse çevirir; eşleşmezse None (örnek atlanır)."""
    g = GROUPS[group]
    canon = g["aliases"].get(raw_name.strip().lower())
    if canon is None:
        return None
    return g["classes"].index(canon)
