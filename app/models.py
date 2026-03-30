from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------

class SectionResult(BaseModel):
    section_name: str
    chunk_id: str


class IngestResponse(BaseModel):
    resume_id: str
    user_id: str
    source_filename: str
    sections_ingested: list[SectionResult]
    name: Optional[str] = None
    skills: list[str] = []
    work_experience_years: Optional[float] = None
    embedding_model: str
    ingested_at: str


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(default=None, ge=1)
    embedding_model: Optional[str] = None


class CandidateResult(BaseModel):
    user_id: str
    resume_id: str
    source_filename: str
    similarity_score: Optional[float] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    work_experience_years: Optional[float] = None
    skills: list[str] = []
    objectives: Optional[str] = None
    matched_sections: list[str] = []
    match_type: str = "rds"  # "rds" | "vector"
    match_reason: Optional[str] = None       # why this result is being shown
    matched_chunk_text: Optional[str] = None # the actual section text used for matching


class QueryLogs(BaseModel):
    user_query: str
    sql_query: str
    sql_matched_count: int
    routing_decision: str  # "rds_only" | "rds_and_vector"
    routing_reason: str
    vector_search_used: bool
    vector_section_used: Optional[str] = None  # "work_experience_text" | "projects" | None
    vector_query: Optional[str] = None
    timing_ms: dict[str, float] = {}  # stage → milliseconds
    op_counts: dict[str, int] = {}    # stage → operation count


class RetrieveResponse(BaseModel):
    query: str
    k_used: int
    candidates: list[CandidateResult]
    logs: Optional[QueryLogs] = None


# ---------------------------------------------------------------------------
# Candidate detail model (full resume view)
# ---------------------------------------------------------------------------

class CandidateDetail(BaseModel):
    resume_id: str
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    objectives: Optional[str] = None
    work_experience_years: Optional[float] = None
    work_experience_text: Optional[str] = None
    projects: Optional[str] = None
    education: Optional[str] = None
    skills: list[str] = []
    achievements: Optional[str] = None
    objectives_chunk_id: Optional[str] = None
    work_experience_text_chunk_id: Optional[str] = None
    projects_chunk_id: Optional[str] = None
    education_chunk_id: Optional[str] = None
    skills_chunk_id: Optional[str] = None
    achievements_chunk_id: Optional[str] = None
    source_filename: str
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[dict] = None


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    resume_id: Optional[str] = None
    source_filename: str
    uploaded_at: str
    name: Optional[str] = None
    work_experience_years: Optional[float] = None
    skills: list[str] = []
