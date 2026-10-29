"""PDF report generation with reportlab."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import config

LIMITATIONS = [
    "The model was trained on a limited public dataset and may not generalise to other scanners, protocols or populations.",
    "Only a single 2-D image is analysed; no clinical history or other imaging sequences are considered.",
    "A high confidence value does not guarantee a correct prediction. Rare tumour types are not covered.",
    "The Grad-CAM heatmap shows image regions that influenced the prediction; it is not a tumour segmentation.",
]


def _fit_image(path: Path, max_w: float, max_h: float) -> Image:
    """Image flowable scaled to fit the box while keeping its aspect ratio."""
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def build_report_pdf(
    record: dict[str, Any], image_path: Path, heatmap_path: Path | None
) -> bytes:
    """Render the report for one stored analysis and return the PDF bytes."""
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=11)
    caption = ParagraphStyle("caption", parent=small, alignment=1, textColor=colors.grey)
    disclaimer_style = ParagraphStyle(
        "disclaimer", parent=styles["BodyText"], backColor=colors.HexColor("#FFF4E5"),
        borderColor=colors.HexColor("#F0B35A"), borderWidth=0.8, borderPadding=6, leading=13,
    )

    label = config.CLASS_LABELS.get(record["predicted_class"], record["predicted_class"])
    story: list[Any] = [
        Paragraph("BrainScan AI - Analysis Report", styles["Title"]),
        Spacer(1, 4 * mm),
    ]

    meta = [
        ["Analysis ID", record["id"]],
        ["Date (UTC)", record["created_at"]],
        ["File name", Paragraph(escape(record["filename"]), styles["BodyText"])],
        ["Model", record["model_version"]],
    ]
    story += [_key_value_table(meta), Spacer(1, 6 * mm)]

    # Images: MRI and (optionally) Grad-CAM side by side.
    cells = [[_fit_image(image_path, 80 * mm, 80 * mm)], [Paragraph("Uploaded MRI", caption)]]
    if heatmap_path is not None:
        cells[0].append(_fit_image(heatmap_path, 80 * mm, 80 * mm))
        cells[1].append(Paragraph("Grad-CAM heatmap (predicted class)", caption))
    story += [Table(cells, hAlign="CENTER", colWidths=[88 * mm] * len(cells[0])), Spacer(1, 6 * mm)]

    story.append(Paragraph("Result", styles["Heading2"]))
    story.append(Paragraph(
        f"Predicted class: <b>{escape(label)}</b> &nbsp; Confidence: <b>{record['confidence'] * 100:.1f}%</b>",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 3 * mm))

    rows = [["Class", "Probability"]]
    for cls in config.CLASSES:
        rows.append([config.CLASS_LABELS[cls], f"{record['probabilities'].get(cls, 0.0) * 100:.2f}%"])
    prob_table = Table(rows, colWidths=[70 * mm, 40 * mm], hAlign="LEFT")
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ]))
    story += [prob_table, Spacer(1, 6 * mm)]

    story.append(Paragraph("Limitations", styles["Heading2"]))
    for item in LIMITATIONS:
        story.append(Paragraph(f"&bull; {escape(item)}", small))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"<b>Disclaimer.</b> {escape(config.DISCLAIMER)}", disclaimer_style))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"BrainScan AI report {record['id']}", author="BrainScan AI",
    )
    doc.build(story)
    return buf.getvalue()


def _key_value_table(rows: list[list[Any]]) -> Table:
    table = Table(rows, colWidths=[35 * mm, 125 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table
