from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.config import get_config
from app.errors import error_response
from app.services.ingestion import ingest_file
from app.utils import slugify_name


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
    kb_name: str | None = Form(default=None),
    chunking_strategy: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    overlap_size: int | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
):
    kb_name = _clean_optional_text(kb_name)
    config = get_config()
    cleaned_strategy = _clean_optional_text(chunking_strategy)
    resolved_chunking_strategy = cleaned_strategy or config.default_chunking_strategy
    embedding_model = _clean_optional_text(embedding_model)
    logger.info(
        "Ingest request received file=%s kb_name=%s chunking_strategy=%s chunk_size=%s overlap_size=%s overwrite=%s embedding_model=%s",
        file.filename,
        kb_name,
        resolved_chunking_strategy,
        chunk_size,
        overlap_size,
        overwrite,
        embedding_model,
    )

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        return error_response(400, "INVALID_FILE_TYPE", f"Unsupported file type '{file_ext}'. Supported: {SUPPORTED_EXTENSIONS}", {"source_filename": file.filename})

    resolved_model = embedding_model or config.embedding_model

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as handle:
        temp_path = Path(handle.name)
        content = await file.read()
        handle.write(content)
    logger.info("Upload stored temporarily at %s", temp_path)

    try:
        response = ingest_file(
            file_name=file.filename,
            file_path=str(temp_path),
            kb_name=kb_name,
            chunking_strategy=resolved_chunking_strategy,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            embedding_model=resolved_model,
            overwrite=overwrite,
        )
        logger.info("Ingest completed kb_name=%s strategies=%s", response.kb_name, [item.strategy_name for item in response.strategies_processed])
        return response.model_dump()
    except FileExistsError as exc:
        return error_response(
            409,
            "DUPLICATE_ENTRY",
            "Active chunks already exist for this file, strategy, and model. Pass overwrite=true to replace them.",
            {
                "source_filename": file.filename,
                "chunking_strategy": resolved_chunking_strategy,
                "embedding_model": resolved_model,
                "kb_name": slugify_name(kb_name) if kb_name else None,
            },
        )
    except LookupError:
        return error_response(400, "NO_EXTRACTABLE_TEXT", "No extractable text was found in the PDF.", {"source_filename": file.filename})
    except ValueError as exc:
        message = str(exc)
        if "password-protected" in message.lower() or "encrypted" in message.lower():
            return error_response(400, "INVALID_FILE_TYPE", "PDF is encrypted or password-protected.", {"source_filename": file.filename})
        if "chunking strategy" in message.lower():
            return error_response(400, "INVALID_CHUNKING_STRATEGY", message)
        return error_response(400, "INGESTION_ERROR", message, {"source_filename": file.filename})
    except Exception as exc:
        return error_response(500, "INGESTION_ERROR", str(exc), {"source_filename": file.filename})
    finally:
        logger.info("Cleaning up temporary file %s", temp_path)
        if temp_path.exists():
            temp_path.unlink()
