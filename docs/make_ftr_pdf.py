"""FTR markdown → PDF (reportlab, şablon biçimine uygun).

Biçim (D-2/şablon): Arial 12 gövde, Arial(-Bold) 14 başlık, 1.15 satır aralığı, iki-yana-yaslı,
kenar boşlukları üst 2.8cm / alt-sağ-sol 2.5cm; Kapak + İçindekiler ayrı sayfa; figürler gömülü.
Arial TTF sistemde varsa kaydedilir; yoksa Helvetica (Arial-metrik eş) kullanılır.

Kullanım:
  python docs/make_ftr_pdf.py --md docs/FTR_icerik.md --out teslim/FTR_GONDERILECEK.pdf \
      --takim "Nankatsu" --takim_id 985007 --basvuru_id <ID>
"""
from __future__ import annotations

import argparse
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")

# --- Arial kaydet (varsa), yoksa Helvetica ---
BODY, BOLD = "Helvetica", "Helvetica-Bold"
for base, bold, paths in [
    ("Arial", "Arial-Bold",
     ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]),
]:
    bolds = ["/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
    rp = next((p for p in paths if os.path.exists(p)), None)
    bp = next((p for p in bolds if os.path.exists(p)), None)
    if rp and bp:
        try:
            pdfmetrics.registerFont(TTFont(base, rp))
            pdfmetrics.registerFont(TTFont(bold, bp))
            BODY, BOLD = base, bold
        except Exception:
            pass

LEAD = 12 * 1.15  # 1.15 satır aralığı
ST = {
    "body": ParagraphStyle("body", fontName=BODY, fontSize=12, leading=LEAD, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=14, leading=16, spaceBefore=12, spaceAfter=6,
                         textColor=colors.HexColor("#1a3c5e")),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=4,
                         textColor=colors.HexColor("#244")),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=12, leading=LEAD, alignment=TA_LEFT,
                             leftIndent=14, spaceAfter=3, bulletIndent=4),
    "cap": ParagraphStyle("cap", fontName=BODY, fontSize=9.5, leading=11, alignment=TA_CENTER,
                          textColor=colors.HexColor("#555"), spaceAfter=8),
    "cover_t": ParagraphStyle("ct", fontName=BOLD, fontSize=20, leading=26, alignment=TA_CENTER),
    "cover_s": ParagraphStyle("cs", fontName=BODY, fontSize=13, leading=20, alignment=TA_CENTER),
}


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def _md_table(block_lines):
    rows = []
    for ln in block_lines:
        if re.match(r"^\s*\|?\s*[-:|\s]+\|?\s*$", ln) and set(ln.replace("|", "").strip()) <= set("-: "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append([Paragraph(_inline(c), ST["body"]) for c in cells])
    if not rows:
        return None
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("FONTNAME", (0, 0), (-1, 0), BOLD),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _fig(name, caption):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return []
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(p).getSize()
    w = min(15.5 * cm, iw * 0.0264583 * cm / 0.0264583)  # px→pt handled by reportlab; cap width
    w = min(15.5 * cm, iw)
    scale = (15.5 * cm) / iw if iw > 15.5 * cm else 1.0
    return [Spacer(1, 6), Image(p, width=iw * scale, height=ih * scale),
            Paragraph(caption, ST["cap"])]


def build(md_path, out_path, takim, takim_id, basvuru_id):
    with open(md_path, encoding="utf-8") as f:
        md = f.read()

    story = []
    # --- Kapak ---
    story += [Spacer(1, 4 * cm),
              Paragraph("5G &amp; YAPAY ZEKA İLE AKILLI YOL GÜVENLİĞİ YARIŞMASI", ST["cover_t"]),
              Spacer(1, 0.6 * cm), Paragraph("FİNAL TASARIM RAPORU", ST["cover_t"]),
              Spacer(1, 2.5 * cm),
              Paragraph(f"Takım Adı: <b>{takim}</b>", ST["cover_s"]),
              Paragraph(f"Takım ID: <b>{takim_id}</b>", ST["cover_s"]),
              Paragraph(f"Başvuru ID: <b>{basvuru_id}</b>", ST["cover_s"]),
              Paragraph("Proje: <b>RoadGuard / teknofestv3</b>", ST["cover_s"]),
              PageBreak()]
    # --- İçindekiler ---
    story += [Paragraph("İçindekiler", ST["h2"]), Spacer(1, 0.3 * cm)]
    for t in ["1. Proje Özeti", "2. Veri Seti Oluşturulması",
              "3. Yapay Zekâ Çözümü (3.1 Problem · 3.2 Mimari · 3.3 Detaylar)",
              "4. Çözümün Sınanması", "5. Kaynakça"]:
        story.append(Paragraph(t, ST["body"]))
    story.append(PageBreak())

    # --- Gövde (markdown ayrıştırma) ---
    lines = md.splitlines()
    i = 0
    para_buf = []

    def flush_para():
        if para_buf:
            story.append(Paragraph(_inline(" ".join(para_buf).strip()), ST["body"]))
            para_buf.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.strip() in ("---", "***"):
            flush_para(); i += 1; continue
        if ln.startswith("## "):
            flush_para(); story.append(Paragraph(_inline(ln[3:].strip()), ST["h2"]))
            # şekil enjeksiyonu
            low = ln.lower()
            if "mimari" in low or "3.2" in low:
                story += _fig("mimari.png", "Şekil 1: Sistem mimarisi — video → results.json")
            i += 1; continue
        if ln.startswith("### "):
            flush_para(); story.append(Paragraph(_inline(ln[4:].strip()), ST["h3"])); i += 1; continue
        if ln.startswith("# "):
            flush_para(); story.append(Paragraph(_inline(ln[2:].strip()), ST["h2"])); i += 1; continue
        if ln.strip().startswith("|") and "|" in ln.strip()[1:]:
            flush_para()
            blk = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                blk.append(lines[i]); i += 1
            t = _md_table(blk)
            if t: story += [t, Spacer(1, 6)]
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            flush_para()
            story.append(Paragraph(_inline(re.sub(r"^\s*[-*]\s+", "", ln)), ST["bullet"], bulletText="•"))
            i += 1; continue
        if ln.strip() == "":
            flush_para()
            if "4. Çözümün Sınanması" in "".join(story_titles(story)[-1:]) if False else False:
                pass
            i += 1; continue
        para_buf.append(ln.strip()); i += 1
    flush_para()
    # Şekil 2'yi sınanma bölümüne ekleyemediysek sona ekle (garanti)
    story += _fig("map_bar.png", "Şekil 2: Held-out mAP@0.5 karşılaştırma (val≠test)")

    doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=2.8 * cm, bottomMargin=2.5 * cm,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            title="FTR — teknofestv3", author=takim)
    doc.build(story)
    print("PDF yazıldı:", out_path, "| font:", BODY)


def story_titles(story):
    return [getattr(s, "text", "") for s in story if hasattr(s, "text")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(HERE, "FTR_icerik.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "teslim", "FTR_GONDERILECEK.pdf"))
    ap.add_argument("--takim", default="Nankatsu")
    ap.add_argument("--takim_id", default="985007")
    ap.add_argument("--basvuru_id", default="-")
    a = ap.parse_args()
    build(a.md, a.out, a.takim, a.takim_id, a.basvuru_id)
