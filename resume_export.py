"""Build an ATS-readable PDF resume from the facts in a saved analysis."""

from __future__ import annotations

import re
import unicodedata
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer


INK = colors.HexColor("#17243A")
MUTED = colors.HexColor("#506078")
ACCENT = colors.HexColor("#254E78")
RULE = colors.HexColor("#C9D5E2")

SECTION_ALIASES = {
    "SUMMARY": "summary", "PROFESSIONAL SUMMARY": "summary", "OBJECTIVE": "summary",
    "PROFILE": "summary", "PROFESSIONAL PROFILE": "summary",
    "EDUCATION": "education", "PROJECTS": "projects", "ACADEMIC PROJECTS": "projects",
    "EXPERIENCE": "experience", "WORK EXPERIENCE": "experience",
    "PROFESSIONAL EXPERIENCE": "experience", "EMPLOYMENT HISTORY": "experience",
    "WORK HISTORY": "experience", "INTERNSHIPS": "experience", "INTERNSHIP": "experience",
    "TECHNICAL SKILLS": "skills", "SKILLS": "skills", "ACHIEVEMENTS & ACTIVITIES": "achievements",
    "ACHIEVEMENTS AND ACTIVITIES": "achievements", "CERTIFICATIONS": "certifications",
    "CERTIFICATES": "certifications", "AWARDS": "achievements", "LANGUAGES": "languages",
    "INTERESTS": "interests", "VOLUNTEER EXPERIENCE": "volunteer",
    "VOLUNTEERING": "volunteer", "COURSEWORK": "coursework",
    "RELEVANT COURSEWORK": "coursework", "TRAINING": "certifications",
}


def _plain(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("\x7f", " ").replace("\u25aa", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t|,;-")


def _split_resume(text: str) -> tuple[str, list[str], dict[str, list[str]]]:
    lines = [_plain(line) for line in text.splitlines()]
    lines = [line for line in lines if line and line not in {"", "•", "▪"}]
    name = lines[0] if lines else "Resume"
    header: list[str] = []
    sections: dict[str, list[str]] = {}
    section: str | None = None
    for line in lines[1:]:
        key = line.rstrip(":").upper()
        if key in SECTION_ALIASES:
            section = SECTION_ALIASES[key]
            sections.setdefault(section, [])
        elif section:
            sections[section].append(line)
        else:
            header.append(line)
    return name, header, sections


def _contact_line(header: list[str]) -> tuple[list[str], str]:
    combined = " | ".join(header)
    email = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", combined, re.I)
    phone = re.search(r"\+?\d[\d ()-]{7,}\d", combined)
    contacts = []
    if email:
        contacts.append(email.group(0))
    if phone:
        contacts.append(re.sub(r"\s+", " ", phone.group(0)).strip())
    address = combined
    if email:
        address = address.replace(email.group(0), " ")
    if phone:
        address = address.replace(phone.group(0), " ")
    address = re.sub(r"\s+", " ", address).strip(" |,;")
    location = ""
    if address:
        segments = [part.strip() for part in re.split(r"[,|;]+", address) if part.strip()]
        # Keep a useful local area while dropping the street and house address.
        location = ", ".join(segments[-2:]) if len(segments) >= 2 else segments[-1]
        location = re.sub(r"\b\d{5,6}\b", "", location)
        location = re.sub(r"\s+", " ", location).strip(" ,")
    return contacts, location


def _skills(sections: dict[str, list[str]]) -> list[str]:
    items = []
    for line in sections.get("skills", []):
        clean = re.sub(r"^[^:]{1,32}:\s*", "", line)
        for item in re.split(r"[,;|•▪]+\s*", clean):
            item = _plain(item)
            if item and item.lower() not in {existing.lower() for existing in items}:
                items.append(item)
    return items


def _source_ordered_skills(items: list[str], matched: list[str]) -> list[str]:
    """Prioritize job-matched skills while keeping the resume's original wording."""
    matched_terms = [item.lower().strip() for item in matched]

    def priority(item: str) -> tuple[int, int]:
        lowered = item.lower()
        return (0 if any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lowered) for term in matched_terms) else 1, items.index(item))

    return sorted(items, key=priority)


def _verified_suggestions(lines: object, source: str) -> list[str]:
    """Keep generated resume wording only when it is copied from source text."""
    if not isinstance(lines, list):
        return []
    source_normalized = " ".join(_plain(source).lower().split())
    accepted = []
    for line in lines:
        if not isinstance(line, str) or not line.strip():
            continue
        cleaned = _plain(line)
        candidate = " ".join(cleaned.lower().split())
        if len(candidate) >= 24 and candidate in source_normalized and cleaned not in accepted:
            accepted.append(cleaned)
    return accepted[:6]


def _projects(lines: list[str]) -> list[tuple[str, list[str]]]:
    projects: list[tuple[str, list[str]]] = []
    title = ""
    bullets: list[str] = []
    for line in lines:
        # PDF list glyphs often extract as U+007F; the source project names are short
        # standalone lines and the descriptive sentences follow them.
        if len(line) < 75 and not re.search(r"[.!?]$", line) and not line.lower().startswith(("developed ", "built ", "implemented ", "researching ", "researched ")):
            if title:
                projects.append((title, bullets))
            title, bullets = line, []
        else:
            line = re.sub(r"^[\s•▪-]+", "", line)
            if line:
                bullets.append(line)
    if title:
        projects.append((title, bullets))
    return projects


def _rewrite_key(value: str) -> str:
    """Normalize extracted bullet text so reviewed suggestions match PDF bullets."""
    value = re.sub(r"^[\s•▪*-]+", "", _plain(value))
    return " ".join(value.casefold().split())


def _looks_like_position(line: str) -> bool:
    return len(line) <= 110 and (
        " | " in line
        or bool(re.search(r"\b(?:19|20)\d{2}\b|\b(?:present|current)\b", line, re.I))
    )


def make_resume_pdf(analysis: dict) -> BytesIO:
    """Create an ATS-readable resume with only user-reviewed role-specific edits."""
    source = analysis.get("resume_text", "")
    name, header, sections = _split_resume(source)
    contacts, location = _contact_line(header)
    role = _plain(analysis.get("job_title", "Target role")) or "Target role"

    base = getSampleStyleSheet()
    styles = {
        "name": ParagraphStyle("ResumeName", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=25, leading=29, alignment=TA_LEFT, textColor=INK, spaceAfter=2),
        "role": ParagraphStyle("ResumeRole", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=13, alignment=TA_LEFT, textColor=ACCENT, spaceAfter=5),
        "contact": ParagraphStyle("ResumeContact", parent=base["Normal"], fontName="Helvetica", fontSize=8.7, leading=12, alignment=TA_LEFT, textColor=MUTED, spaceAfter=3),
        "section": ParagraphStyle("ResumeSection", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=ACCENT, spaceBefore=12, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("ResumeBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=3),
        "small": ParagraphStyle("ResumeSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK, spaceAfter=2.5),
        "position": ParagraphStyle("ResumePosition", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=9.5, leading=12.5, textColor=INK, spaceBefore=4, spaceAfter=2, keepWithNext=True),
        "bullet": ParagraphStyle("ResumeBullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, leftIndent=12, firstLineIndent=-8, textColor=INK, spaceAfter=3),
    }

    def paragraph(text: str, style: str = "body") -> Paragraph:
        return Paragraph(escape(text).replace("\n", "<br/>"), styles[style])

    story = [paragraph(name, "name"), paragraph(role.upper(), "role")]
    contact_text = "  |  ".join(contacts + ([location] if location else []))
    if contact_text:
        story.append(paragraph(contact_text, "contact"))
    story.extend([Spacer(1, 5), HRFlowable(width="100%", thickness=1.3, color=ACCENT), Spacer(1, 2)])

    summary_lines = sections.get("summary", [])
    projects = _projects(sections.get("projects", []))
    experience = sections.get("experience", [])
    report = analysis.get("report", {})
    draft = report.get("tailored_resume", {}) if isinstance(report.get("tailored_resume"), dict) else {}
    matched = report.get("matched_skills", [])
    original_skills = _source_ordered_skills(_skills(sections), matched)
    draft_skills = draft.get("skills", [])
    if isinstance(draft_skills, list):
        draft_skills = [_plain(item) for item in draft_skills if isinstance(item, str) and item.strip()]
    else:
        draft_skills = []
    approved_draft = draft.get("user_approved") is True
    skill_items = draft_skills if approved_draft and draft_skills else original_skills
    profile = draft.get("summary", "").strip() if approved_draft else " ".join(summary_lines).strip()
    replacements = {}
    if approved_draft and isinstance(draft.get("approved_rewrites"), list):
        for item in draft["approved_rewrites"]:
            if isinstance(item, dict) and item.get("source") and item.get("suggestion"):
                replacements[_rewrite_key(item["source"])] = _plain(item["suggestion"])

    def heading(label: str) -> None:
        story.append(paragraph(label.upper(), "section"))
        story.append(HRFlowable(width="100%", thickness=.45, color=RULE, spaceAfter=5))

    if profile:
        heading("Professional Summary")
        story.append(paragraph(profile, "body"))
    if skill_items:
        heading("Technical Skills")
        story.append(paragraph("  ·  ".join(skill_items), "small"))
    if experience:
        heading("Experience")
        for line in experience:
            clean = re.sub(r"^[\s•▪*-]+", "", line).strip()
            replacement = replacements.get(_rewrite_key(clean))
            if _looks_like_position(clean):
                story.append(paragraph(clean, "position"))
            else:
                story.append(Paragraph("• " + escape(replacement or clean), styles["bullet"]))
    if projects:
        heading("Projects")
        for project, bullets in projects:
            story.append(paragraph(project, "position"))
            for bullet in bullets:
                replacement = replacements.get(_rewrite_key(bullet))
                story.append(Paragraph("• " + escape(replacement or bullet), styles["bullet"]))
    if sections.get("education"):
        heading("Education")
        for line in sections["education"]:
            story.append(paragraph(line, "small"))
    if sections.get("certifications"):
        heading("Certifications")
        for line in sections["certifications"]:
            story.append(Paragraph("• " + escape(line), styles["bullet"]))
    if sections.get("achievements"):
        heading("Achievements and Activities")
        for line in sections["achievements"]:
            story.append(Paragraph("• " + escape(line), styles["bullet"]))
    if sections.get("languages"):
        heading("Languages")
        story.append(paragraph(" · ".join(sections["languages"]), "small"))
    if sections.get("coursework"):
        heading("Relevant Coursework")
        story.append(paragraph(" · ".join(sections["coursework"]), "small"))
    if sections.get("volunteer"):
        heading("Volunteer Experience")
        for line in sections["volunteer"]:
            story.append(paragraph(line, "small"))

    output = BytesIO()

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(document.leftMargin, 0.48 * inch, letter[0] - document.rightMargin, 0.48 * inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(letter[0] - document.rightMargin, 0.31 * inch, f"{document.page}")
        canvas.restoreState()

    pdf = SimpleDocTemplate(
        output, pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.58 * inch, bottomMargin=0.58 * inch,
        title=f"{name} - {role} Resume", author="CareerPilot AI",
    )
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    output.seek(0)
    return output
