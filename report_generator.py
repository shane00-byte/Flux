"""
report_generator.py
====================
Builds the three output formats for a completed triage: the on-screen
HTML card, a plain-text summary, and a downloadable PDF slip.
Kept separate from app.py so report formatting can change without
touching the Streamlit control flow.
"""

from __future__ import annotations

import html
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from triage_engine import SEVERITY, TriageResult

def safe_pdf_text(text: str) -> str:
    """Substitutes unsupported Unicode glyphs with '?' to prevent FPDF crashes."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def _esc(value) -> str:
    """HTML-escape any patient-entered text before it goes into markdown(unsafe_allow_html=True)."""
    return html.escape(str(value))


def build_html_report(patient: dict, department: str, result: TriageResult, timestamp: str) -> str:
    sev = SEVERITY[result.severity]
    guardrail_html = (
        f'<p style="color:{sev["color"]};"><strong>Guardrail:</strong> {_esc(result.guardrail_reason)}</p>'
        if result.guardrail_triggered else ""
    )
    return f"""
    <div class="triage-report-card" style="border-left: 6px solid {sev['color']};">
        <h2 style="margin:0 0 6px 0;">Flux AI Clinical Triage Report</h2>
        <p class="report-meta">{_esc(timestamp)}</p>
        <hr>
        <p><strong>Patient:</strong> {_esc(patient['name'])} &nbsp;|&nbsp;
           <strong>Age:</strong> {_esc(patient['age'])} &nbsp;|&nbsp;
           <strong>Sex:</strong> {_esc(patient['sex'])}</p>
        <p><strong>Height / Weight:</strong> {_esc(patient['height'])} cm / {_esc(patient['weight'])} kg
           &nbsp;|&nbsp; <strong>BMI:</strong> {patient['bmi']:.1f} kg/m&sup2;</p>
        <p><strong>Address:</strong> {_esc(patient['address']) if patient['address'] else 'N/A'}</p>
        <p><strong>Medical History:</strong> {_esc(patient['history'])}</p>
        <hr>
        <p><strong>Department:</strong> {_esc(department)}</p>
        <p><strong>Finding:</strong>
           <span style="color:{sev['color']}; font-weight:700;">{sev['icon']} {_esc(result.label)}</span></p>
        <p><strong>Confidence:</strong> {result.probability:.1f}% &nbsp;|&nbsp;
           <strong>Priority:</strong>
           <span style="color:{sev['color']}; font-weight:700;">{sev['label']}</span></p>
        {guardrail_html}
        ---
    
    *For clinical decision support only. Not a substitute for professional medical judgment.*
    </div>
    """


def build_text_report(patient: dict, department: str, result: TriageResult, timestamp: str) -> str:
    sev = SEVERITY[result.severity]
    lines = [
        "FLUX AI CLINICAL TRIAGE REPORT",
        f"Generated: {timestamp}",
        "-" * 52,
        f"Patient Name : {patient['name']}",
        f"Age          : {patient['age']} years",
        f"Sex          : {patient['sex']}",
        f"Height/Weight: {patient['height']} cm / {patient['weight']} kg",
        f"BMI          : {patient['bmi']:.1f} kg/m2",
        f"Address      : {patient['address'] or 'N/A'}",
        f"History      : {patient['history']}",
        "-" * 52,
        f"Department   : {department}",
        f"Finding      : {result.label}",
        f"Confidence   : {result.probability:.1f}%",
        f"Priority     : {sev['label']}",
    ]
    if result.guardrail_triggered:
        lines.append(f"Guardrail    : {result.guardrail_reason}")
    lines += ["-" * 52, "For clinical decision support only.", "Explainable AI Triage System"]
    return "\n".join(lines)


def build_pdf_report(patient: dict, department: str, result: TriageResult, timestamp: str) -> bytes:
    sev = SEVERITY[result.severity]
    rgb = tuple(int(sev["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    pdf = FPDF(format=(148, 210))  # A5 in mm, set explicitly to avoid a library warning
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 40, 60)
    pdf.cell(0, 9, "Flux AI Clinical Triage Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, timestamp, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    def row(label, value):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(45, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 7, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    row("Patient:", safe_pdf_text(patient["name"]))
    row("Age / Sex:", f"{patient['age']} yrs / {patient['sex']}")
    row("Height / Weight:", f"{patient['height']} cm / {patient['weight']} kg")
    row("BMI:", f"{patient['bmi']:.1f} kg/m2")
    row("Address:", safe_pdf_text(patient["address"] or "N/A"))
    row("Med History:", safe_pdf_text(patient["history"]))

    pdf.ln(2)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    row("Department:", department)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*rgb)
    pdf.multi_cell(0, 8, f"{sev['label'].upper()}: {result.label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, f"Confidence: {result.probability:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if result.guardrail_triggered:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 6, f"Guardrail: {result.guardrail_reason}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "For clinical decision support only. Not a substitute for "
                         "professional medical judgment.")

    out = pdf.output()
    return bytes(out)


def now_str() -> str:
    return datetime.now().strftime("%d-%b-%Y %H:%M:%S")
