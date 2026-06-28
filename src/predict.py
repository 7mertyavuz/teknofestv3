"""v3 çıkarım orkestratörü — video → D-2 results.json (event-stream'siz, tek-geçiş).

Proven roadguard primitiflerini (tespit+ByteTrack, sürücü-durum 16/8 oylama, plaka
custom-LP+OCR konsensüsü, driver_lock) Pipeline.frames() üzerinden sürer; ÇIKTIYI
D-2 şemasına (src/d2_labels) çevirir. QoD/speed/event-stream/dashboard KULLANILMAZ
(D-2 results.json kapsamı dışı). Araç tip/renk için src/vehicle_attrs.

Akış:
  Pipeline.frames(video) → her kare: pipe.acc.tracks (tam TrackRecord) okunur →
  ana araç + davranış/nesne/yolcu epizotları zaman damgasıyla biriktirilir →
  arac_bilgisi + tespitler kurulur → D-2 doğrulamasından geçer.
DAYANIKLILIK (D-2 §7): tek bozuk kare TÜM koşuyu bitirmez (kare-başına try/except).
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict

from src import vehicle_attrs
from src.d2_labels import (
    DRIVER_FLAG_TO_D2,
    OBJECT_CLASS_TO_D2,
    PLATE_UNREADABLE,
    clamp_conf,
    normalize_color,
    normalize_plate,
    round_time,
    to_ascii_lower,
)

log = logging.getLogger("teknofestv3.predict")

# Epizot ayrımı: aynı etiketin ardışık tespitleri bu boşluktan (sn) fazla ayrıksa
# AYRI olay sayılır (her olay için tek temsilci zaman damgası + tepe güven).
EPISODE_GAP_S = 1.2
# Bir epizodun çıktıya girmesi için min tepe güven (saf gürültüyü ele; düşük-ama-gerçek kalsın).
MIN_EPISODE_CONF = 0.25
# Araç tip/renk örneklemesi için saklanan en büyük-alan kırpık sayısı (oylama robustluğu).
TOP_CROPS = 5
# Yolcu epizodu çıktıya girmek için min kare (geçici/tek-kare yanlış-pozitif yolcuyu ele).
MIN_PASSENGER_FRAMES = 8
# SÜRE BÜTÇESİ (D-2 ≤10dk): işleme bu süreyi aşarsa döngü kesilir, ELDEKİ sonuçla GEÇERLİ
# results.json yazılır (hakem konteyneri 10dk'da öldürür → çıktısız kalmaktansa kısmi-geçerli).
# Tek davranış, ortam-tespiti yok (§5.4). Model yükleme dahil run_inference başından ölçülür.
RUNTIME_BUDGET_S = 540.0


def _collapse_episodes(samples: list[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """[(t, conf), ...] → epizot başına (temsilci_t, tepe_conf, kare_sayısı). t-sıralı."""
    if not samples:
        return []
    samples = sorted(samples, key=lambda x: x[0])
    episodes: list[list[tuple[float, float]]] = [[samples[0]]]
    for t, c in samples[1:]:
        if t - episodes[-1][-1][0] <= EPISODE_GAP_S:
            episodes[-1].append((t, c))
        else:
            episodes.append([(t, c)])
    out = []
    for ep in episodes:
        peak_t, peak_c = max(ep, key=lambda x: x[1])
        out.append((peak_t, peak_c, len(ep)))
    return out


def _seat_label(person: dict, vehicle_bbox) -> str | None:
    """Yolcunun araç kutusu içindeki konumundan D-2 koltuk etiketi (kaba geometri).

    Ön yarı → on_koltuk; arka yarı sol/sağ → arka_koltuk_1/2. Sürücü zaten role=driver gelir.
    """
    b = person.get("bbox")
    if not b or vehicle_bbox is None:
        return None
    px = (b[0] + b[2]) / 2.0
    py = (b[1] + b[3]) / 2.0
    vx1, vy1, vx2, vy2 = vehicle_bbox.x1, vehicle_bbox.y1, vehicle_bbox.x2, vehicle_bbox.y2
    vw = max(1.0, vx2 - vx1)
    vh = max(1.0, vy2 - vy1)
    rx = (px - vx1) / vw
    ry = (py - vy1) / vh
    if ry < 0.55:
        return "on_koltuk"
    return "arka_koltuk_1" if rx < 0.5 else "arka_koltuk_2"


def run_inference(video_path: str, weights_path: str = "/app/weights", max_frames=None) -> dict:
    """Videoyu işleyip D-2 results.json sözlüğü döndürür."""
    from roadguard.config import load_config
    from roadguard.pipeline.pipeline import Pipeline

    weights_dir = weights_path if os.path.isdir(weights_path) else os.path.dirname(weights_path)
    cfg = load_config()
    pipe = Pipeline(cfg)
    type_clf = vehicle_attrs.VehicleTypeClassifier(weights_dir)

    video_id = os.path.basename(video_path)

    veh: dict[int, dict] = defaultdict(
        lambda: {
            "frames": 0,
            "conf_sum": 0.0,
            "area_max": 0.0,
            "class_votes": defaultdict(float),
            "plate_value": None,
            "plate_conf": 0.0,
            "plate_partial": None,
            "top_crops": [],  # [(area, bgr_crop)]
            "last_bbox": None,
        }
    )
    behavior_spans: dict[tuple, list[tuple[float, float]]] = defaultdict(list)  # (tid,kat,etiket)
    object_spans: dict[str, list[tuple[float, float]]] = defaultdict(list)  # d2_label
    # Kişi-merkezli rol takibi (pid → rol sayıları + koltuk oyu + zaman). Rolü kare-kare
    # salınan TEK sürücüyü yanlışlıkla yolcu saymamak için: yolcu = ÇOĞUNLUKLA passenger
    # olan VE sürücü-baskın olmayan kişi.
    person_roles: dict[int, dict] = {}

    def _accumulate(frame, anno) -> None:
        """Tek karenin katkısını birikimlere yazar (kare-başına izole edilir)."""
        fps_ = pipe.fps or 30.0
        t = round_time(anno.frame_id / max(fps_, 1e-6))
        fh, fw = frame.shape[:2]

        for tid, rec in list(pipe.acc.tracks.items()):
            if rec.last_frame != anno.frame_id:
                continue
            bb = rec.bbox
            area = max(0.0, bb.width) * max(0.0, bb.height)
            v = veh[tid]
            v["frames"] += 1
            v["conf_sum"] += float(bb.conf)
            v["class_votes"][rec.vehicle_class or ""] += area
            v["last_bbox"] = bb
            if area > v["area_max"]:
                v["area_max"] = area
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = min(fw, int(bb.x2)), min(fh, int(bb.y2))
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                crop = frame[y1:y2, x1:x2].copy()
                tc = v["top_crops"]
                tc.append((area, crop))
                tc.sort(key=lambda z: z[0], reverse=True)
                del tc[TOP_CROPS:]

            if rec.plate.status == "confirmed" and rec.plate.value:
                v["plate_value"] = rec.plate.value
                v["plate_conf"] = max(v["plate_conf"], clamp_conf(rec.plate.confidence))
            elif rec.plate.partial and not v["plate_value"]:
                v["plate_partial"] = rec.plate.partial

            dconf = rec.driver.confidence or {}
            for flag in rec.driver.active_flags():
                d2 = DRIVER_FLAG_TO_D2.get(flag)
                if d2:
                    behavior_spans[(tid, d2[0], d2[1])].append(
                        (t, clamp_conf(dconf.get(flag, 0.6)))
                    )
            if rec.speed.swerving:
                d2 = DRIVER_FLAG_TO_D2["swerving"]
                behavior_spans[(tid, d2[0], d2[1])].append((t, clamp_conf(0.6)))

        for aux in getattr(pipe.detector, "last_aux", []):
            d2lab = OBJECT_CLASS_TO_D2.get(to_ascii_lower(getattr(aux, "cls", "")))
            if d2lab:
                object_spans[d2lab].append((t, clamp_conf(getattr(aux, "conf", 0.5))))

        for p in anno.persons:
            pid = p.get("track_id")
            if pid is None:
                continue
            role = p.get("role")
            vid = p.get("vehicle_id")
            rec = person_roles.setdefault(
                pid, {"driver": 0, "passenger": 0, "vid": vid,
                      "seats": Counter(), "times": []}
            )
            rec[role] = rec.get(role, 0) + 1
            rec["vid"] = vid
            if role == "passenger":
                vrec = pipe.acc.tracks.get(vid)
                seat = _seat_label(p, vrec.bbox if vrec is not None else None)
                if seat:
                    rec["seats"][seat] += 1
                    rec["times"].append((t, 0.5))

    start_t = time.time()
    budget_hit = False
    try:
        for frame, anno, _events in pipe.frames(video_path, max_frames=max_frames):
            if time.time() - start_t > RUNTIME_BUDGET_S:
                budget_hit = True
                log.warning(
                    "Süre bütçesi (%.0fs) aşıldı, kare %s'te kesiliyor — eldeki sonuçla finalize",
                    RUNTIME_BUDGET_S, getattr(anno, "frame_id", "?"),
                )
                break
            try:
                _accumulate(frame, anno)
            except Exception as fe:  # noqa: BLE001 — tek kare hatası tüm koşuyu bitirmesin
                log.debug("kare %s atlandı: %s", getattr(anno, "frame_id", "?"), fe)
                continue
        if budget_hit:
            log.info("Bütçe-kesimli finalize: %.0fs işlendi", time.time() - start_t)
    except Exception as e:  # noqa: BLE001 — kaynak açma/okuma hatası: kısmi sonuçla devam (D-2 §7)
        log.warning("Çıkarım kaynak hatası (kısmi sonuçla devam): %s", e)
    finally:
        try:
            pipe.close()
        except Exception:  # noqa: BLE001
            pass

    return _build_results(video_id, veh, behavior_spans, object_spans, person_roles, type_clf)


def _build_results(video_id, veh, behavior_spans, object_spans, person_roles, type_clf) -> dict:
    """Birikimlerden D-2 results.json sözlüğünü kurar."""
    tespitler: list[dict] = []

    main_tid = None
    if veh:
        main_tid = max(veh, key=lambda k: (veh[k]["frames"], veh[k]["area_max"]))

    arac = {"tip": None, "plaka": PLATE_UNREADABLE, "renk": None, "confidence_score": 0.0}
    if main_tid is not None:
        v = veh[main_tid]
        best_crop = v["top_crops"][0][1] if v["top_crops"] else None
        stock_cls = max(v["class_votes"], key=v["class_votes"].get) if v["class_votes"] else ""
        tip, tip_conf = type_clf.infer(best_crop, stock_cls)

        color_votes: dict[str, float] = defaultdict(float)
        color_conf_acc = 0.0
        color_n = 0
        for _area, crop in v["top_crops"]:
            c, cc = vehicle_attrs.estimate_color(crop)
            if c:
                color_votes[c] += cc
                color_conf_acc += cc
                color_n += 1
        renk = normalize_color(max(color_votes, key=color_votes.get)) if color_votes else None
        color_conf = (color_conf_acc / color_n) if color_n else 0.0

        plaka = normalize_plate(v["plate_value"]) if v["plate_value"] else PLATE_UNREADABLE
        plate_conf = v["plate_conf"] if plaka != PLATE_UNREADABLE else 0.0
        det_conf = clamp_conf(v["conf_sum"] / max(1, v["frames"]))
        parts = [p for p in (tip_conf, color_conf, plate_conf, det_conf) if p > 0]
        arac = {
            "tip": tip,
            "plaka": plaka,
            "renk": renk,
            "confidence_score": clamp_conf(sum(parts) / len(parts)) if parts else 0.0,
        }

    for (tid, kat, etiket), samples in behavior_spans.items():
        if main_tid is not None and tid != main_tid:
            continue
        for peak_t, peak_c, _n in _collapse_episodes(samples):
            if peak_c >= MIN_EPISODE_CONF:
                tespitler.append(
                    {"zaman_saniye": peak_t, "kategori": kat, "etiket": etiket,
                     "confidence_score": clamp_conf(peak_c)}
                )

    for d2lab, samples in object_spans.items():
        for peak_t, peak_c, _n in _collapse_episodes(samples):
            if peak_c >= MIN_EPISODE_CONF:
                tespitler.append(
                    {"zaman_saniye": peak_t, "kategori": "nesneler", "etiket": d2lab,
                     "confidence_score": clamp_conf(peak_c)}
                )

    # Yolcular: ana araçta EN ÇOK kare gören kişi = SÜRÜCÜ (yolcu değil; driver_lock tek
    # kişiyi bazen passenger etiketler → o FP elenir). Yolcu YALNIZ ikincil, çoğunlukla
    # passenger ve kalıcı (>=MIN_PASSENGER_FRAMES) kişilerden.
    main_persons = {
        pid: rec for pid, rec in person_roles.items()
        if main_tid is None or rec.get("vid") == main_tid
    }
    driver_pid = (
        max(main_persons, key=lambda k: main_persons[k]["driver"] + main_persons[k]["passenger"])
        if main_persons else None
    )
    for pid, rec in main_persons.items():
        if pid == driver_pid:
            continue  # birincil/tek kişi = sürücü
        pf, df = rec.get("passenger", 0), rec.get("driver", 0)
        if pf >= MIN_PASSENGER_FRAMES and pf > df and rec.get("seats"):
            seat = rec["seats"].most_common(1)[0][0]
            episodes = _collapse_episodes(rec.get("times", []))
            if episodes:
                peak_t, peak_c, _n = max(episodes, key=lambda e: e[2])
                tespitler.append(
                    {"zaman_saniye": peak_t, "kategori": "yolcular", "etiket": seat,
                     "confidence_score": clamp_conf(peak_c)}
                )

    tespitler.sort(key=lambda d: (d["zaman_saniye"], d["kategori"], d["etiket"]))
    return {"video_id": video_id, "arac_bilgisi": arac, "tespitler": tespitler}
