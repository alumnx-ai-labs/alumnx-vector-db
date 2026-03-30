from __future__ import annotations

import logging

from fastapi import APIRouter

from app.errors import error_response
from app.models import CandidateDetail
from app.services.store.postgres_store import PostgresStore

router = APIRouter()
logger = logging.getLogger("nexvec.candidates")


@router.get("/candidates/{resume_id}")
def get_candidate(resume_id: str):
    """Fetch full resume details for a candidate by resume_id."""
    try:
        pg = PostgresStore()
        row = pg.get_resume_by_id(resume_id)
        if not row:
            return error_response(404, "NOT_FOUND", f"No candidate found for resume_id={resume_id!r}")

        detail = CandidateDetail(
            resume_id=row["resume_id"],
            user_id=row["user_id"],
            name=row.get("name"),
            email=row.get("email"),
            phone=row.get("phone"),
            location=row.get("location"),
            objectives=row.get("objectives"),
            work_experience_years=row.get("work_experience_years"),
            work_experience_text=row.get("work_experience_text"),
            projects=row.get("projects"),
            education=row.get("education"),
            skills=list(row.get("skills") or []),
            achievements=row.get("achievements"),
            objectives_chunk_id=row.get("objectives_chunk_id"),
            work_experience_text_chunk_id=row.get("work_experience_text_chunk_id"),
            projects_chunk_id=row.get("projects_chunk_id"),
            education_chunk_id=row.get("education_chunk_id"),
            skills_chunk_id=row.get("skills_chunk_id"),
            achievements_chunk_id=row.get("achievements_chunk_id"),
            source_filename=row.get("source_filename", ""),
            created_at=row.get("created_at"),
        )
        return detail.model_dump()
    except Exception as exc:
        logger.exception("Failed to fetch candidate %s", resume_id)
        return error_response(500, "FETCH_ERROR", str(exc))
