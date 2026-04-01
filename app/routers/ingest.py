from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.errors import error_response
from app.services.ingestion import ingest_file
from app.services.store.s3_store import S3Store


router = APIRouter()
logger = logging.getLogger("nexvec.ingest")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "string":
        return None
    return stripped


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    embedding_model: str | None = Form(default=None),
):
    embedding_model = _clean_optional_text(embedding_model)
    logger.info("Ingest request: file=%s embedding_model=%s", file.filename, embedding_model)

    if not file.filename.lower().endswith(".pdf"):
        return error_response(
            400, "INVALID_FILE_TYPE",
            "Only PDF resumes are supported.",
            {"source_filename": file.filename},
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
        temp_path = Path(handle.name)
        handle.write(await file.read())

    # before ingest save to s3
    s3_store = S3Store()
    s3_store.upload_file(str(temp_path), file.filename)

    try:
        response = ingest_file(
            file_name=file.filename,
            file_path=str(temp_path),
            embedding_model=embedding_model,
        )
        logger.info(
            "Ingest completed file=%s resume_id=%s sections=%d",
            file.filename, response.resume_id, len(response.sections_ingested),
        )
        return response.model_dump()
    except LookupError:
        return error_response(
            400, "NO_EXTRACTABLE_TEXT",
            "No extractable text found in the PDF.",
            {"source_filename": file.filename},
        )
    except Exception as exc:
        logger.exception("Ingest failed for file=%s", file.filename)
        return error_response(500, "INGESTION_ERROR", str(exc), {"source_filename": file.filename})
    finally:
        if temp_path.exists():
            temp_path.unlink()
