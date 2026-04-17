import io
import os
import base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Colors ───────────────────────────────────────────
BLUE       = colors.HexColor("#2b6cb0")
LIGHT_BLUE = colors.HexColor("#ebf4ff")
RED        = colors.HexColor("#c53030")
ORANGE     = colors.HexColor("#c05621")
GREEN      = colors.HexColor("#276749")
GRAY       = colors.HexColor("#718096")
LIGHT_GRAY = colors.HexColor("#f7fafc")
DARK       = colors.HexColor("#1a202c")
WHITE      = colors.white


def build_pdf(
    patient_name: str,
    patient_age:  str,
    patient_gender: str,
    patient_id:   str,
    doctor:       str,
    notes:        str,
    filename:     str,
    predictions:  dict,   # { pathology: {score, cam_base64} }
    orig_b64:     str,    # original xray base64 (may be empty)
) -> bytes:
    """
    Generate a medical PDF report.
    Returns raw PDF bytes.
    """

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Styles ───────────────────────────────────────
    title_style = ParagraphStyle("title",
        fontSize=20, fontName="Helvetica-Bold",
        textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)

    sub_style = ParagraphStyle("sub",
        fontSize=10, fontName="Helvetica",
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=2)

    section_style = ParagraphStyle("section",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=BLUE, spaceBefore=14, spaceAfter=6)

    body_style = ParagraphStyle("body",
        fontSize=10, fontName="Helvetica",
        textColor=DARK, leading=16)

    small_gray = ParagraphStyle("small",
        fontSize=8, fontName="Helvetica",
        textColor=GRAY, alignment=TA_CENTER)

    disclaimer_style = ParagraphStyle("disclaimer",
        fontSize=8, fontName="Helvetica-Oblique",
        textColor=GRAY, alignment=TA_CENTER,
        borderColor=ORANGE, borderWidth=0.5,
        borderPadding=6, spaceBefore=10)

    # ── Header ───────────────────────────────────────
    logo_path = os.path.join(os.path.dirname(__file__), "..", "static", "logo.png")
    if os.path.exists(logo_path):
        logo = RLImage(logo_path, width=6*cm, height=2.5*cm)
        logo.hAlign = "CENTER"
        story.append(logo)
        story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Chest X-Ray Analysis Report", sub_style))
    story.append(Paragraph("DenseNet121 · 14 Pulmonary Pathologies · Grad-CAM", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=12))

    # ── Patient info table ────────────────────────────
    now = datetime.now().strftime("%d %B %Y · %H:%M")
    story.append(Paragraph("Patient Information", section_style))

    info_data = [
        ["Full Name",   patient_name,  "Date",        now],
        ["Age",         patient_age,   "Gender",      patient_gender],
        ["Patient ID",  patient_id,    "Physician",   doctor],
        ["File",        filename,      "Notes",       notes or "—"],
    ]

    info_table = Table(info_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 6*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), LIGHT_BLUE),
        ("BACKGROUND",  (2,0), (2,-1), LIGHT_BLUE),
        ("TEXTCOLOR",   (0,0), (0,-1), BLUE),
        ("TEXTCOLOR",   (2,0), (2,-1), BLUE),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUND", (0,0), (-1,-1), [WHITE, LIGHT_GRAY]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0,0), (-1,-1), 6),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4*cm))

    # ── X-Ray images ─────────────────────────────────
    story.append(Paragraph("Radiographic Images", section_style))

    img_cells = []

    # Original
    if orig_b64:
        try:
            raw = base64.b64decode(
                orig_b64.split(",")[1] if "," in orig_b64 else orig_b64
            )
            orig_buf = io.BytesIO(raw)
            orig_img = RLImage(orig_buf, width=7.5*cm, height=7.5*cm)
            img_cells.append([orig_img, Paragraph("Original X-Ray", small_gray)])
        except Exception:
            img_cells.append(["[Image unavailable]", "Original X-Ray"])

    # Top CAM
    entries = list(predictions.items())
    if entries:
        top_name, top_data = entries[0]
        try:
            cam_raw = base64.b64decode(top_data["cam_base64"])
            cam_buf = io.BytesIO(cam_raw)
            cam_img = RLImage(cam_buf, width=7.5*cm, height=7.5*cm)
            img_cells.append([
                cam_img,
                Paragraph(f"Grad-CAM · {top_name}", small_gray)
            ])
        except Exception:
            img_cells.append(["[CAM unavailable]", f"Grad-CAM · {top_name}"])

    if img_cells:
        # Each cell: [image, caption] stacked
        row_imgs = [cell[0] for cell in img_cells]
        row_caps = [cell[1] for cell in img_cells]
        img_table = Table(
            [row_imgs, row_caps],
            colWidths=[8.5*cm] * len(img_cells)
        )
        img_table.setStyle(TableStyle([
            ("ALIGN",   (0,0), (-1,-1), "CENTER"),
            ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(img_table)

    story.append(Spacer(1, 0.3*cm))

    # ── Scores table ─────────────────────────────────
    story.append(Paragraph("Pathology Probability Scores", section_style))

    score_data = [["Pathology", "Score", "Probability", "Risk Level"]]
    for name, d in predictions.items():
        pct   = round(d["score"] * 100, 1)
        risk  = "HIGH"   if pct >= 50 else ("MODERATE" if pct >= 35 else "LOW")
        color_map = {"HIGH": RED, "MODERATE": ORANGE, "LOW": GREEN}
        score_data.append([
            name,
            f"{d['score']:.4f}",
            f"{pct}%",
            Paragraph(f'<font color="{color_map[risk].hexval()}">{risk}</font>',
                      ParagraphStyle("r", fontSize=9, fontName="Helvetica-Bold"))
        ])

    score_table = Table(score_data, colWidths=[6*cm, 3*cm, 3.5*cm, 4.5*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUND", (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING",    (0,0), (-1,-1), 6),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Diagnosis ────────────────────────────────────
    story.append(Paragraph("Final Diagnostic Summary", section_style))

    top_name  = entries[0][0]
    top_score = round(entries[0][1]["score"] * 100, 1)
    high_list = [n for n, d in predictions.items() if d["score"] >= 0.5]
    confidence = "high" if top_score >= 55 else "moderate"

    diag_text = (
        f"<b>Primary finding:</b> {top_name} ({top_score}%) — {confidence} confidence.<br/>"
        f"<b>Elevated pathologies (≥50%):</b> {', '.join(high_list) if high_list else 'None'}.<br/>"
        f"<b>Recommendation:</b> Clinical correlation and radiologist review advised."
    )

    diag_table = Table([[Paragraph(diag_text, body_style)]], colWidths=[17*cm])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT_BLUE),
        ("GRID",       (0,0), (-1,-1), 1, BLUE),
        ("PADDING",    (0,0), (-1,-1), 10),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Disclaimer ───────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠️ This report is generated by an AI system and is intended to assist qualified medical "
        "professionals only. It does not constitute a medical diagnosis and must be reviewed and "
        "validated by a licensed radiologist or physician before any clinical decision is made.",
        disclaimer_style
    ))

    doc.build(story)
    return buf.getvalue()