from __future__ import annotations

import logging
import uuid

import numpy as np

from app.config import get_config
from app.models import IngestResponse, StrategyResult
from app.services.chunking.registry import get_chunker_registry
from app.services.embedding.embedder import GeminiEmbedder
from app.services.pdf_extractor import extract_pdf_pages
from app.services.store.duplicate_checker import find_duplicate_rows
from app.services.store.jsonl_store import JSONLStore
from app.utils import now_ist, now_ist_iso, slugify_name


SUPPORTED_CHUNKING_STRATEGIES = {"fixed_length", "paragraph", "both"}
logger = logging.getLogger("nexvec.ingestion")


def _normalise_vector(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm == 0:
        return [0.0 for _ in vector]
    return (array / norm).astype(float).tolist()


def _resolve_kb_name(source_filename: str, provided_kb_name: str | None) -> str:
    if provided_kb_name:
        return slugify_name(provided_kb_name)
    stem = slugify_name(source_filename)
    timestamp = now_ist().strftime("%Y%m%d_%H%M")
    return f"{stem}_{timestamp}"


def _chunk_page_text(chunker, page_number: int, text: str) -> list[tuple[int, str]]:
    chunks = chunker.split(text)
    return [(page_number, chunk) for chunk in chunks]


def ingest_file(
    file_name: str,
    file_path: str,
    kb_name: str | None,
    chunking_strategy: str,
    chunk_size: int | None,
    overlap_size: int | None,
    embedding_model: str | None,
    overwrite: bool,
) -> IngestResponse:
    config = get_config()
    store = JSONLStore()
    resolved_kb_name = _resolve_kb_name(file_name, kb_name)
    logger.info("Resolved kb_name=%s for source=%s", resolved_kb_name, file_name)

    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError("Unsupported chunking strategy requested.")

    target_path = store.kb_path(resolved_kb_name)
    existing_rows = store.read_rows(resolved_kb_name) if target_path.exists() else []
    # If both strategies are requested, we need to consider each strategy independently.
    strategy_names = ["fixed_length", "paragraph"] if chunking_strategy == "both" else [chunking_strategy]
    active_duplicates_by_strategy = {
        strategy: find_duplicate_rows(
            existing_rows,
            source_filename=file_name,
            chunking_strategy=strategy,
            embedding_model=embedding_model or config.embedding_model,
        )
        for strategy in strategy_names
    }

    if not overwrite:
        duplicate_hit = next((strategy for strategy, rows in active_duplicates_by_strategy.items() if rows), None)
        if duplicate_hit:
            logger.info(
                "Duplicate ingestion rejected source=%s kb_name=%s strategy=%s model=%s",
                file_name,
                resolved_kb_name,
                duplicate_hit,
                embedding_model or config.embedding_model,
            )
            raise FileExistsError(
                f"Active chunks already exist for file={file_name}, strategy={duplicate_hit}, model={embedding_model or config.embedding_model}"
            )

    if overwrite and existing_rows:
        logger.info("Overwrite enabled; deactivating matching rows in kb_name=%s", resolved_kb_name)
        deactivated_at = now_ist_iso()
        for row in existing_rows:
            if row.get("source_filename") == file_name and row.get("chunking_strategy") in strategy_names and row.get("embedding_model") == (embedding_model or config.embedding_model) and row.get("is_active"):
                row["is_active"] = False
                row["deactivated_at"] = deactivated_at
        store.update_rows(resolved_kb_name, existing_rows)
        existing_rows = store.read_rows(resolved_kb_name)

    from pathlib import Path
    import mimetypes

    embedder = GeminiEmbedder(embedding_model or config.embedding_model)
    created_at = now_ist_iso()
    strategies_processed: list[StrategyResult] = []
    new_rows: list[dict] = []
    ext = Path(file_name).suffix.lower()

    if ext == ".pdf":
        pages = extract_pdf_pages(file_path)
        if not pages:
            raise LookupError("NO_EXTRACTABLE_TEXT")
        logger.info("Extracted %s text pages from %s", len(pages), file_name)

        effective_chunk_size = chunk_size or config.chunk_size
        effective_overlap_size = overlap_size if overlap_size is not None else config.overlap_size
        if effective_overlap_size >= effective_chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

        chunkers = get_chunker_registry(effective_chunk_size, effective_overlap_size)

        for strategy_name in strategy_names:
            logger.info("Starting chunking strategy=%s", strategy_name)
            chunker = chunkers[strategy_name]
            indexed_chunks: list[tuple[int, str]] = []
            for page in pages:
                indexed_chunks.extend(_chunk_page_text(chunker, page.page_number, page.text))

            chunk_texts = [chunk_text for _, chunk_text in indexed_chunks]
            logger.info("Generating embeddings strategy=%s chunk_count=%s model=%s", strategy_name, len(chunk_texts), embedder.model)
            vectors = embedder.embed_texts(chunk_texts)
            if len(vectors) != len(indexed_chunks):
                raise RuntimeError("Embedding vector count does not match chunk count.")
            strategy_rows: list[dict] = []
            for index, ((page_number, chunk_text), vector) in enumerate(zip(indexed_chunks, vectors)):
                row = {
                    "chunk_id": str(uuid.uuid4()),
                    "kb_name": resolved_kb_name,
                    "source_filename": file_name,
                    "chunking_strategy": strategy_name,
                    "chunk_index": index,
                    "page_number": page_number,
                    "chunk_text": chunk_text,
                    "embedding_model": embedder.model,
                    "embedding_vector": vector,
                    "normalised_vector": _normalise_vector(vector),
                    "vector_size": config.vector_size,
                    "chunk_size_used": effective_chunk_size,
                    "overlap_size_used": effective_overlap_size if strategy_name == "fixed_length" else None,
                    "is_active": True,
                    "created_at": created_at,
                    "deactivated_at": None,
                }
                strategy_rows.append(row)
            new_rows.extend(strategy_rows)
            logger.info("Prepared %s rows for strategy=%s", len(strategy_rows), strategy_name)
            strategies_processed.append(
                StrategyResult(
                    strategy_name=strategy_name,
                    chunk_count=len(strategy_rows),
                    embedding_model=embedder.model,
                    vector_size=config.vector_size,
                    overwritten=overwrite and bool(active_duplicates_by_strategy.get(strategy_name)),
                )
            )
    else:
        mime_type, _ = mimetypes.guess_type(file_name)
        if ext == ".txt":
            result = embedder.embed_text_file(file_path)
        elif ext in {".png", ".jpg", ".jpeg"}:
            result = embedder.embed_image(file_path, mime_type or "image/jpeg")
        elif ext in {".mp3", ".wav"}:
            result = embedder.embed_audio(file_path, mime_type or "audio/mpeg")
        elif ext in {".mp4"}:
            result = embedder.embed_video(file_path, mime_type or "video/mp4")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        description = result["description"]

        if not description.strip():
            raise LookupError("NO_EXTRACTABLE_TEXT")

        effective_chunk_size = chunk_size or config.chunk_size
        effective_overlap_size = overlap_size if overlap_size is not None else config.overlap_size
        if effective_overlap_size >= effective_chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

        chunkers = get_chunker_registry(effective_chunk_size, effective_overlap_size)

        for strategy_name in strategy_names:
            logger.info("Starting chunking strategy=%s for media file=%s", strategy_name, file_name)
            chunker = chunkers[strategy_name]
            # Treat the description as a single page of text
            indexed_chunks: list[tuple[int, str]] = _chunk_page_text(chunker, 1, description)

            chunk_texts = [chunk_text for _, chunk_text in indexed_chunks]
            logger.info(
                "Generating embeddings strategy=%s chunk_count=%s model=%s",
                strategy_name, len(chunk_texts), embedder.model,
            )
            vectors = embedder.embed_texts(chunk_texts)
            if len(vectors) != len(indexed_chunks):
                raise RuntimeError("Embedding vector count does not match chunk count.")

            strategy_rows: list[dict] = []
            for index, ((page_number, chunk_text), vector) in enumerate(zip(indexed_chunks, vectors)):
                row = {
                    "chunk_id": str(uuid.uuid4()),
                    "kb_name": resolved_kb_name,
                    "source_filename": file_name,
                    "chunking_strategy": strategy_name,
                    "chunk_index": index,
                    "page_number": page_number,
                    "chunk_text": chunk_text,
                    "embedding_model": embedder.model,
                    "embedding_vector": vector,
                    "normalised_vector": _normalise_vector(vector),
                    "vector_size": config.vector_size,
                    "chunk_size_used": effective_chunk_size,
                    "overlap_size_used": effective_overlap_size if strategy_name == "fixed_length" else None,
                    "is_active": True,
                    "created_at": created_at,
                    "deactivated_at": None,
                }
                strategy_rows.append(row)
            new_rows.extend(strategy_rows)
            logger.info("Prepared %s rows for strategy=%s", len(strategy_rows), strategy_name)
            strategies_processed.append(
                StrategyResult(
                    strategy_name=strategy_name,
                    chunk_count=len(strategy_rows),
                    embedding_model=embedder.model,
                    vector_size=config.vector_size,
                    overwritten=overwrite and bool(active_duplicates_by_strategy.get(strategy_name)),
                )
            )

    store.write_rows(resolved_kb_name, new_rows)
    logger.info("Wrote %s rows to kb_name=%s", len(new_rows), resolved_kb_name)
    return IngestResponse(
        kb_name=resolved_kb_name,
        source_filename=file_name,
        strategies_processed=strategies_processed,
        ingested_at=created_at,
    )
