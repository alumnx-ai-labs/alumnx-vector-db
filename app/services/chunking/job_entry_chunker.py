from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

logger = logging.getLogger("nexvec.chunker")

# Text shorter than this is a single chunk — no LLM call needed
_MIN_CHUNK_LENGTH = 250

_CHUNK_PROMPT = """\
You are splitting a resume section into individual semantic units for a vector search system.

Rules:
- For work_experience_text: one chunk per job, internship, or role.
- For projects: one chunk per distinct project.
- Each chunk must be self-contained — keep all related details (title, company, dates, responsibilities) together.
- Minimum chunk length: 60 characters. Merge very short entries with the nearest one.
- If the text has only one entry or cannot be clearly split, return a single-element array.
- Return ONLY a valid JSON array of strings. No markdown, no explanation, nothing else.

Section type: {section}

Text:
\"\"\"{text}\"\"\"
"""

# Fallback: split on blank lines between entries if LLM fails
_BLANK_LINE_RE = re.compile(r"\n{2,}")


def _heuristic_split(text: str) -> list[str]:
    """Split on blank lines; merge very short pieces."""
    parts = [p.strip() for p in _BLANK_LINE_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < 60:
            merged[-1] = merged[-1] + "\n" + part
        else:
            merged.append(part)
    return merged if merged else [text]


def chunk_section(section: str, text: str) -> list[str]:
    """
    Split a resume section into semantic chunks using Gemini Flash.

    Returns a list of chunk texts. Always returns at least one element.
    Falls back to heuristic blank-line splitting if the LLM call fails.
    """
    text = text.strip()
    if not text:
        return []

    # Short text — not worth splitting
    if len(text) < _MIN_CHUNK_LENGTH:
        return [text]

    # Truncate extremely long sections to avoid token waste
    truncated = text[:6000] if len(text) > 6000 else text

    try:
        client = genai.Client()
        prompt = _CHUNK_PROMPT.format(section=section, text=truncated)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw).strip()

        chunks = json.loads(raw)
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("LLM returned empty or non-list")

        cleaned = [str(c).strip() for c in chunks if str(c).strip()]
        if not cleaned:
            raise ValueError("All chunks empty after cleaning")

        logger.info("Chunker: section=%s → %d chunks (LLM)", section, len(cleaned))
        return cleaned

    except Exception as exc:
        logger.warning(
            "Chunker LLM failed for section=%s (%s) — using heuristic split", section, exc
        )
        fallback = _heuristic_split(text)
        logger.info("Chunker: section=%s → %d chunks (heuristic)", section, len(fallback))
        return fallback
