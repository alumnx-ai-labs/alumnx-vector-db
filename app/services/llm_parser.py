from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types

logger = logging.getLogger("nexvec.llm_parser")

_PARSE_PROMPT = """\
You are an expert resume parser. Your job is to extract ALL content from the resume \
and map every piece of it into exactly these 7 fields. Nothing may be left out.

Return ONLY valid JSON — no markdown, no explanation — with exactly this structure:
{
  "name": "<full name or null>",
  "email": "<email address or null>",
  "phone": "<digits only, no spaces or symbols, or null>",
  "location": "<city, country or null>",
  "objectives": "<text>",
  "work_experience_years": <float or null>,
  "work_experience_text": "<text>",
  "projects": "<text>",
  "education": "<text>",
  "skills": ["<skill>", ...],
  "achievements": "<text>"
}

Mapping rules — be flexible, map by CONTENT not by heading name:

  objectives:
    Map here: career objective, professional summary, about me, profile, personal statement,
    introduction. If none exist, write a 1-sentence summary inferred from the resume.

  work_experience_years:
    Sum of all work durations from explicit dates (internships count). null if no dates found.

  work_experience_text:
    Map here: full-time jobs, internships, part-time work, freelance, training,
    industrial training, apprenticeships, work placements — any professional experience.
    If the resume has internships but no full-time jobs, put them here.

  projects:
    Map here: personal projects, academic projects, capstone, final year project,
    open source contributions, case studies, coursework projects.

  education:
    Map here: degrees, diplomas, courses, certifications, online courses, bootcamps,
    schools, colleges, universities, GPA, grades.

  skills:
    Every technical skill, tool, framework, language, platform, library mentioned
    anywhere in the resume. Extract from all sections.

  achievements:
    Map here: awards, honours, hackathons, competitions, publications, patents,
    extracurricular activities, leadership roles, volunteer work, certifications,
    positions of responsibility, anything notable not covered above.
    If the resume has certifications or extracurricular activities, put them here.

Critical rules:
- EVERY piece of content from the resume must appear in at least one field
- Only use null for work_experience_years if no dates exist to calculate from
- objectives, work_experience_text, projects, education, achievements must NEVER be null
  if there is any content that could reasonably map there — use your judgement
- skills defaults to [] if truly none found
- Return ONLY the JSON object, no markdown fences

Resume text:
\"\"\"
{text}
\"\"\"\
"""


@dataclass
class ParsedResume:
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    objectives: str | None = None
    work_experience_years: float | None = None
    work_experience_text: str | None = None
    projects: str | None = None
    education: str | None = None
    skills: list[str] = field(default_factory=list)
    achievements: str | None = None


def parse_resume(full_text: str) -> ParsedResume:
    """
    Call Gemini Flash to extract the 7 fixed sections and identity fields
    from raw resume text. Returns a flat ParsedResume dataclass.

    Raises RuntimeError if the LLM response cannot be parsed.
    """
    client = genai.Client()
    before, after = _PARSE_PROMPT.split("{text}", 1)
    prompt = before + full_text.strip() + after

    logger.info("Calling LLM to parse resume (text_len=%d)", len(full_text))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    raw = response.text.strip()
    logger.debug("LLM raw response: %s", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

    def _safe_float(val) -> float | None:
        try:
            return round(float(val), 1) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _safe_str(val) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    def _safe_str_list(val) -> list[str]:
        if not isinstance(val, list):
            return []
        return [str(v).strip() for v in val if v]

    result = ParsedResume(
        name=_safe_str(parsed.get("name")),
        email=_safe_str(parsed.get("email")),
        phone=_safe_str(parsed.get("phone")),
        location=_safe_str(parsed.get("location")),
        objectives=_safe_str(parsed.get("objectives")),
        work_experience_years=_safe_float(parsed.get("work_experience_years")),
        work_experience_text=_safe_str(parsed.get("work_experience_text")),
        projects=_safe_str(parsed.get("projects")),
        education=_safe_str(parsed.get("education")),
        skills=_safe_str_list(parsed.get("skills")),
        achievements=_safe_str(parsed.get("achievements")),
    )

    logger.info(
        "Parsed resume: name=%r email=%r exp_years=%s skills=%d",
        result.name, result.email, result.work_experience_years, len(result.skills),
    )
    return result
