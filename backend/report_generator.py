"""
report_generator.py — PDF breach report using reportlab.
Called by POST /api/generate-report in main.py.
"""

import io
import json
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── colour palette ────────────────────────────────────────────
C_RED    = colors.HexColor("#dc2626")
C_ORANGE = colors.HexColor("#ea580c")
C_YELLOW = colors.HexColor("#ca8a04")
C_GREEN  = colors.HexColor("#16a34a")
C_DARK   = colors.HexColor("#1e293b")
C_MID    = colors.HexColor("#475569")
C_LIGHT  = colors.HexColor("#f1f5f9")
C_WHITE  = colors.white
C_ACCENT = colors.HexColor("#6366f1")

RISK_COLOUR = {
    "CRITICAL": C_RED,
    "HIGH":     C_ORANGE,
    "MEDIUM":   C_YELLOW,
    "LOW":      C_GREEN,
}

MITRE_MAP = {
    "reconnaissance":       ("TA0043", "Reconnaissance",         ["T1595", "T1590", "T1589"]),
    "lateral_movement":     ("TA0008", "Lateral Movement",       ["T1021", "T1550", "T1563"]),
    "privilege_escalation": ("TA0004", "Privilege Escalation",   ["T1068", "T1548", "T1134"]),
    "exfiltration":         ("TA0010", "Exfiltration",           ["T1041", "T1048", "T1567"]),
    "initial_access":       ("TA0001", "Initial Access",         ["T1190", "T1566", "T1078"]),
    "execution":            ("TA0002", "Execution",              ["T1059", "T1203", "T1569"]),
}

RECOMMENDATIONS = {
    "CRITICAL": [
        "Immediately isolate affected systems and revoke all active credentials.",
        "Engage incident response team and begin forensic investigation.",
        "Reset all privileged accounts and rotate API keys / secrets.",
        "Deploy additional honeytokens across high-value assets.",
        "Notify stakeholders and initiate breach notification procedures.",
    ],
    "HIGH": [
        "Suspend the compromised token and rotate related credentials.",
        "Review access logs for the 72 h window before the alert.",
        "Increase monitoring sensitivity on affected network segments.",
        "Deploy canary tokens in adjacent directories and services.",
        "Audit all lateral-movement paths identified in the attack graph.",
    ],
    "MEDIUM": [
        "Investigate the triggered token and correlate with SIEM events.",
        "Review user permissions and apply least-privilege where possible.",
        "Schedule a threat-hunt for related IOCs within 48 h.",
        "Refresh the honeytoken deployment in the affected zone.",
    ],
    "LOW": [
        "Log and monitor — no immediate action required.",
        "Validate that no sensitive data was exposed near the token.",
        "Run the next scheduled security review ahead of schedule.",
    ],
}


def _styles():
    base = getSampleStyleSheet()
    extra = {
        "Cover":      ParagraphStyle("Cover",      fontSize=26, textColor=C_WHITE,  alignment=TA_CENTER, spaceAfter=6,  fontName="Helvetica-Bold"),
        "CoverSub":   ParagraphStyle("CoverSub",   fontSize=12, textColor=C_LIGHT,  alignment=TA_CENTER, spaceAfter=4,  fontName="Helvetica"),
        "H1":         ParagraphStyle("H1",         fontSize=14, textColor=C_DARK,   spaceBefore=12, spaceAfter=4,  fontName="Helvetica-Bold"),
        "H2":         ParagraphStyle("H2",         fontSize=11, textColor=C_ACCENT, spaceBefore=8,  spaceAfter=3,  fontName="Helvetica-Bold"),
        "Body":       ParagraphStyle("Body",       fontSize=9,  textColor=C_MID,    spaceAfter=3,  fontName="Helvetica", leading=14),
        "Code":       ParagraphStyle("Code",       fontSize=8,  textColor=C_DARK,   fontName="Courier", backColor=C_LIGHT, spaceAfter=2, leftIndent=6),
        "Risk":       ParagraphStyle("Risk",       fontSize=18, textColor=C_WHITE,  alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "Footer":     ParagraphStyle("Footer",     fontSize=7,  textColor=C_MID,    alignment=TA_CENTER),
    }
    return {**{k: base[k] for k in base.byName}, **extra}


def _section(title, styles) -> list:
    return [
        Spacer(1, 4 * mm),
        HRFlowable(width="100%", thickness=1, color=C_ACCENT),
        Paragraph(title, styles["H1"]),
    ]


def generate_report(data: dict) -> bytes:
    """
    Build a PDF breach report and return it as bytes.

    Expected keys in `data`:
        token_id, token_type, token_value,
        attacker_ip, triggered_at,           # may be None/empty
        attack_stages (list[str]),
        risk_score (float 0-1),
        entropy, entropy_ratio, disc_score, authenticity,
        attribution (str),
        attack_graph_summary (dict, optional),
        threat_attribution_summary (dict, optional),
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
        title="Honeytoken Breach Report",
    )

    styles  = _styles()
    story   = []
    W       = A4[0] - 40*mm          # usable width

    # ── risk level ──────────────────────────────────────────
    raw_risk  = float(data.get("risk_score", 0.5))
    if raw_risk >= 0.85:   risk_level = "CRITICAL"
    elif raw_risk >= 0.65: risk_level = "HIGH"
    elif raw_risk >= 0.40: risk_level = "MEDIUM"
    else:                  risk_level = "LOW"
    risk_colour = RISK_COLOUR[risk_level]

    # ══════════════════════════════════════════════════════════
    # COVER BLOCK
    # ══════════════════════════════════════════════════════════
    cover_table = Table(
        [[
            Paragraph("🔐 HONEYTOKEN BREACH REPORT", styles["Cover"]),
            Paragraph(f"Risk Level", styles["CoverSub"]),
        ]],
        colWidths=[W * 0.72, W * 0.28],
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, 0), C_DARK),
        ("BACKGROUND",  (1, 0), (1, 0), risk_colour),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (1, 0), (1, 0), "CENTER"),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",  (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
    ]))

    risk_cell = Table(
        [[Paragraph(risk_level, styles["Risk"])],
         [Paragraph(f"{raw_risk*100:.0f} / 100", styles["CoverSub"])]],
    )
    risk_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_colour),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(Table(
        [[Paragraph("🔐  HONEYTOKEN BREACH REPORT", styles["Cover"]),
          risk_cell]],
        colWidths=[W * 0.70, W * 0.30],
        style=[
            ("BACKGROUND",  (0, 0), (0, 0), C_DARK),
            ("BACKGROUND",  (1, 0), (1, 0), risk_colour),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING", (0, 0), (0, 0), 14),
        ],
    ))
    story.append(Spacer(1, 3*mm))

    # generated-at line
    gen_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(
        f"<font color='#6366f1'><b>Generated:</b></font>  {gen_at} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<font color='#6366f1'><b>System:</b></font>  AI Honeytoken Factory v5",
        styles["Body"],
    ))
    story.append(Spacer(1, 2*mm))

    # ══════════════════════════════════════════════════════════
    # 1. INCIDENT OVERVIEW
    # ══════════════════════════════════════════════════════════
    story += _section("1. Incident Overview", styles)

    triggered_at = data.get("triggered_at") or datetime.utcnow().isoformat()
    if hasattr(triggered_at, "isoformat"):
        triggered_at = triggered_at.isoformat()

    overview_data = [
        ["Field", "Value"],
        ["Token ID",       str(data.get("token_id",    "N/A"))],
        ["Token Type",     str(data.get("token_type",  "N/A")).upper()],
        ["Attacker IP",    str(data.get("attacker_ip", "Unknown"))],
        ["Triggered At",   str(triggered_at)[:19].replace("T", " ")],
        ["Attribution",    str(data.get("attribution", "Unknown"))],
    ]
    overview_table = Table(overview_data, colWidths=[W*0.28, W*0.72])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("BACKGROUND",    (0, 1), (-1, -1), C_LIGHT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(overview_table)

    # ══════════════════════════════════════════════════════════
    # 2. TOKEN QUALITY METRICS
    # ══════════════════════════════════════════════════════════
    story += _section("2. Token Quality Metrics", styles)

    def pct(v): return f"{float(v or 0)*100:.1f}%"

    metrics_data = [
        ["Metric",          "Value",                    "Target",   "Status"],
        ["Shannon Entropy",  f"{data.get('entropy', 0):.4f} bits",  "≥ 4.5",   "✅" if float(data.get("entropy", 0)) >= 4.5 else "⚠️"],
        ["Entropy Ratio",    pct(data.get("entropy_ratio", 0)),      "≥ 90%",   "✅" if float(data.get("entropy_ratio", 0)) >= 0.90 else "⚠️"],
        ["Disc Score",       pct(data.get("disc_score",    0)),      "≥ 90%",   "✅" if float(data.get("disc_score",    0)) >= 0.90 else "⚠️"],
        ["Authenticity",     pct(data.get("authenticity",  0)),      "≥ 90%",   "✅" if float(data.get("authenticity",  0)) >= 0.90 else "⚠️"],
    ]
    m_table = Table(metrics_data, colWidths=[W*0.35, W*0.25, W*0.20, W*0.20])
    m_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR",      (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(m_table)

    # ══════════════════════════════════════════════════════════
    # 3. ATTACK STAGES
    # ══════════════════════════════════════════════════════════
    story += _section("3. Detected Attack Stages", styles)

    stages = data.get("attack_stages") or ["lateral_movement"]
    if isinstance(stages, str):
        stages = [s.strip() for s in stages.split(",")]

    stage_rows = [["Stage", "MITRE Tactic", "Tactic ID", "Technique IDs"]]
    for s in stages:
        key  = s.lower().replace(" ", "_")
        info = MITRE_MAP.get(key, ("—", s.replace("_", " ").title(), ["—"]))
        stage_rows.append([
            s.replace("_", " ").title(),
            info[1],
            info[0],
            ", ".join(info[2]),
        ])

    s_table = Table(stage_rows, colWidths=[W*0.22, W*0.28, W*0.15, W*0.35])
    s_table.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",      (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(s_table)

    # ══════════════════════════════════════════════════════════
    # 4. ATTACK GRAPH SUMMARY (optional)
    # ══════════════════════════════════════════════════════════
    ag = data.get("attack_graph_summary") or {}
    if ag:
        story += _section("4. Attack Graph Intelligence (Module 2)", styles)
        ag_rows = [["Property", "Value"]]
        for k, v in {
            "Risk Level":           ag.get("risk_level", "—"),
            "Max Anomaly Score":    f"{ag.get('max_anomaly_score', 0):.4f}",
            "Mean Anomaly Score":   f"{ag.get('mean_anomaly_score', 0):.4f}",
            "Attack Windows":       str(ag.get("attack_windows_count", 0)),
            "Dominant Stage":       str(ag.get("dominant_stage", "—")).replace("_", " ").title(),
            "Total Events":         str(ag.get("total_events", 0)),
        }.items():
            ag_rows.append([k, v])
        ag_table = Table(ag_rows, colWidths=[W*0.40, W*0.60])
        ag_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ]))
        story.append(ag_table)

        # MITRE mappings from attack graph
        mitre = ag.get("mitre_mappings") or []
        if mitre:
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph("MITRE ATT&CK Mappings", styles["H2"]))
            for m in mitre[:6]:
                story.append(Paragraph(
                    f"• <b>{m.get('tactic_id','')}</b> — {m.get('tactic','')} "
                    f"| Techniques: {', '.join(m.get('techniques', [])[:3])}",
                    styles["Body"],
                ))

    # ══════════════════════════════════════════════════════════
    # 5. THREAT ATTRIBUTION SUMMARY (optional)
    # ══════════════════════════════════════════════════════════
    ta = data.get("threat_attribution_summary") or {}
    if ta:
        story += _section("5. Threat Attribution (Module 4)", styles)
        attrib = ta.get("campaign_attribution") or {}
        if attrib:
            ta_rows = [["Property", "Value"]]
            for k, v in {
                "Predicted Actor":  str(attrib.get("predicted_actor", "—")),
                "Confidence":       f"{float(attrib.get('confidence', 0))*100:.1f}%",
                "Method":           str(attrib.get("method", "—")).title(),
            }.items():
                ta_rows.append([k, v])
            ta_table = Table(ta_rows, colWidths=[W*0.40, W*0.60])
            ta_table.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",       (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING",    (0, 0), (-1, -1), 8),
                ("TOPPADDING",     (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ]))
            story.append(ta_table)

    # ══════════════════════════════════════════════════════════
    # 6. RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════
    story += _section("6. Recommendations", styles)
    for i, rec in enumerate(RECOMMENDATIONS.get(risk_level, []), 1):
        story.append(Paragraph(f"{i}. {rec}", styles["Body"]))

    # token value (redacted)
    tv = str(data.get("token_value", ""))
    if tv and len(tv) > 8:
        redacted = tv[:6] + "…" + tv[-4:]
    else:
        redacted = "***REDACTED***"
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Triggered Token (redacted):", styles["H2"]))
    story.append(Paragraph(redacted, styles["Code"]))

    # ══════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "CONFIDENTIAL — AI Honeytoken Factory v5 | "
        "This report was generated automatically and should be reviewed by a security analyst.",
        styles["Footer"],
    ))

    doc.build(story)
    return buf.getvalue()
