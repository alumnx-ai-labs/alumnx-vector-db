from __future__ import annotations

"""Tests for app.services.chunking.job_entry_chunker.chunk_section."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.chunking.job_entry_chunker import chunk_section, _heuristic_split


# ---------------------------------------------------------------------------
# Test 1: Short text skips LLM entirely
# ---------------------------------------------------------------------------

def test_chunk_section_short_text_skips_llm():
    short_text = "Senior Engineer at Acme."  # well under 250 chars
    with patch("app.services.chunking.job_entry_chunker.genai") as mock_genai:
        result = chunk_section("work_experience_text", short_text)

    mock_genai.Client.assert_not_called()
    assert result == [short_text]


# ---------------------------------------------------------------------------
# Test 2: LLM returns valid JSON array → use it
# ---------------------------------------------------------------------------

def test_chunk_section_llm_returns_json_array():
    long_text = "A" * 300  # over 250 chars to trigger LLM path

    mock_response = MagicMock()
    mock_response.text = '["Job A entry", "Job B entry"]'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.chunking.job_entry_chunker.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = chunk_section("work_experience_text", long_text)

    assert result == ["Job A entry", "Job B entry"]
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Test 3: LLM raises exception → heuristic fallback
# ---------------------------------------------------------------------------

def test_chunk_section_llm_failure_falls_back_to_heuristic():
    entry_a = "Job at Company A.\nBuilt stuff." + " " * 10
    entry_b = "Job at Company B.\nDid more things." + " " * 10
    long_text = entry_a + "\n\n" + entry_b
    # Pad to ensure it's > 250 chars
    long_text = long_text.ljust(300)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("LLM exploded")

    with patch("app.services.chunking.job_entry_chunker.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = chunk_section("work_experience_text", long_text)

    # Should fall back to heuristic (blank-line split)
    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# Test 4: Empty string returns empty list
# ---------------------------------------------------------------------------

def test_chunk_section_empty_string_returns_empty_list():
    result = chunk_section("work_experience_text", "")
    assert result == []


# ---------------------------------------------------------------------------
# Test 5: Heuristic split on blank-line separated entries
# ---------------------------------------------------------------------------

def test_chunk_section_heuristic_split_blank_lines():
    block_a = "Senior Engineer at TechCorp 2021-2024.\nBuilt scalable microservices."
    block_b = "Junior Engineer at StartupXYZ 2019-2021.\nWorked on backend systems."
    text = block_a + "\n\n" + block_b

    result = _heuristic_split(text)

    assert len(result) == 2
    assert block_a in result[0]
    assert block_b in result[1]


# ---------------------------------------------------------------------------
# Test 6: LLM returns empty array → fall back to heuristic
# ---------------------------------------------------------------------------

def test_chunk_section_llm_empty_array_falls_back():
    long_text = "B" * 300  # over 250 chars

    mock_response = MagicMock()
    mock_response.text = "[]"  # empty array

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.services.chunking.job_entry_chunker.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = chunk_section("projects", long_text)

    # Empty array triggers fallback → heuristic returns at least the full text
    assert isinstance(result, list)
    assert len(result) >= 1
