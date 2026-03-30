from __future__ import annotations

import logging
import time

import numpy as np

from app.config import get_config
from app.models import CandidateResult, QueryLogs, RetrieveRequest, RetrieveResponse
from app.services.embedding.embedder import GeminiEmbedder
from app.services.ingestion import UNIVERSAL_VECTOR_STORE
from app.services.llm_query import classify_and_generate_sql
from app.services.store.hnsw_store import get_hnsw_store
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore

logger = logging.getLogger("nexvec.retrieval")


def _ms(t0: float) -> float:
    """Milliseconds elapsed since t0 (perf_counter)."""
    return round((time.perf_counter() - t0) * 1000, 2)


def _benchmark_vector_formats(
    vfs: VectorFileStore,
    kb_name: str,
    raw_query_vector: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Read all vector storage formats and return (timings_ms, op_counts).
    Observability benchmark — intentionally kept for early-stage diagnostics.
    """
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}

    t0 = time.perf_counter()
    all_vecs, all_ids = vfs.read(kb_name)
    timings["vector_npy_ms"] = _ms(t0)
    counts["vector_npy_count"] = len(all_ids)

    t0 = time.perf_counter()
    jsonl_records = vfs.read_jsonl(kb_name)
    timings["vector_jsonl_ms"] = _ms(t0)
    counts["vector_jsonl_count"] = len(jsonl_records)

    t0 = time.perf_counter()
    gz_records = vfs.read_normalized_gz(kb_name)
    timings["vector_json_gz_ms"] = _ms(t0)
    counts["vector_json_gz_count"] = len(gz_records)

    t0 = time.perf_counter()
    index = vfs.read_index(kb_name)
    timings["vector_index_ms"] = _ms(t0)
    counts["vector_index_count"] = len(index)

    t0 = time.perf_counter()
    raw_vecs, raw_ids = vfs.read_raw(kb_name)
    if raw_query_vector is not None and len(raw_ids) > 0:
        _ = (raw_vecs @ raw_query_vector).tolist()
    timings["vector_raw_ms"] = _ms(t0)
    counts["vector_raw_count"] = len(raw_ids)

    return timings, counts


def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    """
    RDS-first retrieval with LLM-driven routing and HyDE query enhancement.

    Flow:
      1. LLM classifies the query → SQL + routing + hypothetical_doc (HyDE)
      2. Execute SQL → Postgres returns matching resume_ids
      3a. RDS-only: fetch full resume rows, return ranked as SQL matched them
      3b. RDS + Vector:
           - Embed hypothetical_doc (if present) or raw query as retrieval_document
           - Score all chunks for SQL-filtered resumes via HNSW / exact KNN
           - Aggregate best score per resume_id
      4. Deduplicate by user_id — one result per person
      5. Return ranked candidates + query logs
    """
    config = get_config()
    k = request.k or config.knn_k
    pg = PostgresStore()
    t_total = time.perf_counter()
    timing: dict[str, float] = {}
    op_counts: dict[str, int] = {
        "llm_sql_gen_count": 1,
        "embedding_count": 1,
        "k_requested": k,
        "total_active_resumes_count": pg.get_active_resume_count(),
    }

    if not request.query.strip():
        raise ValueError("EMPTY_QUERY")

    embedding_model = request.embedding_model or config.embedding_model
    logger.info("Retrieve: query=%r k=%s model=%s", request.query, k, embedding_model)

    # ── Step 1: Classify query → SQL + routing + HyDE doc ─────────────
    sql_failed = False
    sql = ""
    needs_vector = False
    routing_reason = ""
    hypothetical_doc: str | None = None

    try:
        t0 = time.perf_counter()
        classification = classify_and_generate_sql(request.query)
        timing["llm_sql_gen_ms"] = _ms(t0)

        sql = classification.sql
        needs_vector = classification.needs_vector
        routing_reason = classification.reason
        hypothetical_doc = classification.hypothetical_doc

        t0 = time.perf_counter()
        resume_ids = pg.execute_sql_query(sql)
        timing["postgres_sql_ms"] = _ms(t0)
        op_counts["postgres_sql_count"] = len(resume_ids)

        logger.info(
            "SQL returned %d resume_ids | needs_vector=%s | hyde=%s | reason=%s",
            len(resume_ids), needs_vector, bool(hypothetical_doc), routing_reason,
        )
    except Exception as exc:
        logger.warning("SQL classification/execution failed (%s), falling back to full scan", exc)
        resume_ids = []
        sql_failed = True
        routing_reason = f"SQL failed ({exc}), falling back to full scan with vector search"
        needs_vector = True

    sql_matched_count = len(resume_ids)

    if resume_ids:
        t0 = time.perf_counter()
        resume_rows = pg.get_resumes_by_ids(resume_ids)
        timing["postgres_fetch_ms"] = _ms(t0)
        op_counts["postgres_fetch_count"] = len(resume_rows)
    elif sql_failed:
        t0 = time.perf_counter()
        resume_rows = pg.get_all_active_resumes()
        timing["postgres_fetch_ms"] = _ms(t0)
        op_counts["postgres_fetch_count"] = len(resume_rows)
    elif needs_vector:
        logger.info(
            "SQL returned 0 for semantic query=%r — falling back to full-scan vector search",
            request.query,
        )
        routing_reason = routing_reason + " (SQL keyword too narrow — vector search on full pool)"
        t0 = time.perf_counter()
        resume_rows = pg.get_all_active_resumes()
        timing["postgres_fetch_ms"] = _ms(t0)
        op_counts["postgres_fetch_count"] = len(resume_rows)
    else:
        logger.info("SQL filter returned no matches for query=%r", request.query)
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=0,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
            timing_ms={**timing, "total_ms": _ms(t_total)},
            op_counts=op_counts,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=[], logs=logs)

    if not resume_rows:
        logger.info("No active resumes found")
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=0,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
            timing_ms={**timing, "total_ms": _ms(t_total)},
            op_counts=op_counts,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=[], logs=logs)

    resume_rows_by_id = {r["resume_id"]: r for r in resume_rows}

    # ── Step 2: RDS-only path (no vector search) ───────────────────────
    if not needs_vector:
        logger.info("RDS-only path: returning %d candidates without vector ranking", len(resume_rows))

        seen_users: set[str] = set()
        candidates: list[CandidateResult] = []

        ordered_ids = [rid for rid in resume_ids if rid in resume_rows_by_id]
        remaining = [r["resume_id"] for r in resume_rows if r["resume_id"] not in set(ordered_ids)]
        all_ordered = ordered_ids + remaining

        for rid in all_ordered:
            row = resume_rows_by_id.get(rid, {})
            uid = row.get("user_id", rid)
            if uid in seen_users:
                continue
            seen_users.add(uid)
            candidates.append(CandidateResult(
                user_id=uid,
                resume_id=rid,
                source_filename=row.get("source_filename", ""),
                similarity_score=None,
                name=row.get("name"),
                email=row.get("email"),
                phone=row.get("phone"),
                location=row.get("location"),
                work_experience_years=row.get("work_experience_years"),
                skills=list(row.get("skills") or []),
                objectives=row.get("objectives"),
                matched_sections=[],
                match_type="rds",
                match_reason="SQL structured filter matched query criteria against skills, experience, location, or education",
                matched_chunk_text=row.get("work_experience_text") or row.get("projects") or row.get("objectives"),
            ))
            if len(candidates) == k:
                break

        op_counts["resumes_before_dedup_count"] = len(all_ordered)
        op_counts["candidates_returned_count"] = len(candidates)
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=sql_matched_count,
            routing_decision="rds_only",
            routing_reason=routing_reason,
            vector_search_used=False,
            timing_ms={**timing, "total_ms": _ms(t_total)},
            op_counts=op_counts,
        )
        logger.info("RDS-only complete: %d candidates", len(candidates))
        return RetrieveResponse(query=request.query, k_used=k, candidates=candidates, logs=logs)

    # ── Step 3: RDS + Vector path ──────────────────────────────────────
    logger.info("RDS + Vector path: scoring %d SQL-filtered resumes", len(resume_rows))

    # Load all chunks for the filtered resume pool from the chunks table
    all_resume_ids = [r["resume_id"] for r in resume_rows]
    t0 = time.perf_counter()
    chunk_records = pg.get_chunks_for_resume_ids(all_resume_ids)
    timing["chunks_fetch_ms"] = _ms(t0)

    chunk_to_resume: dict[str, str] = {c["chunk_id"]: c["resume_id"] for c in chunk_records}
    chunk_to_section: dict[str, str] = {c["chunk_id"]: c["section"] for c in chunk_records}
    chunk_to_text: dict[str, str | None] = {c["chunk_id"]: c.get("chunk_text") for c in chunk_records}
    op_counts["chunks_loaded_count"] = len(chunk_records)

    # ── Embed HyDE hypothetical_doc or raw query ───────────────────────
    # HyDE: embed as retrieval_document (same distribution as stored vectors)
    # Raw query: embed as retrieval_query
    embedder = GeminiEmbedder(embedding_model)
    text_to_embed = hypothetical_doc or request.query
    embed_as_doc = bool(hypothetical_doc)

    t0 = time.perf_counter()
    if embed_as_doc:
        raw_query_vector = np.asarray(embedder.embed_texts([text_to_embed])[0], dtype=np.float32)
        logger.info("HyDE: embedding hypothetical_doc (%d chars) as retrieval_document", len(text_to_embed))
    else:
        raw_query_vector = np.asarray(embedder.embed_query(text_to_embed), dtype=np.float32)
    timing["embedding_ms"] = _ms(t0)
    op_counts["hyde_used"] = 1 if embed_as_doc else 0

    norm = np.linalg.norm(raw_query_vector)
    query_vector = (raw_query_vector / norm) if norm > 0 else raw_query_vector

    # Observability benchmark across all storage formats
    vfs = VectorFileStore()
    format_timings, format_counts = _benchmark_vector_formats(vfs, UNIVERSAL_VECTOR_STORE, raw_query_vector)
    timing.update(format_timings)
    op_counts.update(format_counts)
    op_counts["vector_dims_count"] = int(query_vector.shape[0])

    # ── Score chunks via HNSW or exact KNN ────────────────────────────
    allowed_chunk_ids = list(chunk_to_resume.keys())
    op_counts["vectors_scored_count"] = len(allowed_chunk_ids)

    score_by_chunk: dict[str, float] = {}
    if allowed_chunk_ids:
        hnsw = get_hnsw_store(config.vector_size)
        t0 = time.perf_counter()
        results = hnsw.search_filtered(query_vector, allowed_chunk_ids, k=len(allowed_chunk_ids))
        timing["cosine_scoring_ms"] = _ms(t0)
        score_by_chunk = {cid: score for cid, score in results}

    if not score_by_chunk:
        logger.warning(
            "Vector search found no scoring chunks for %d SQL-matched resumes — "
            "falling back to RDS result order", len(resume_rows)
        )
        seen_users_fb: set[str] = set()
        fallback_candidates: list[CandidateResult] = []
        ordered_fb = [rid for rid in resume_ids if rid in resume_rows_by_id]
        remaining_fb = [r["resume_id"] for r in resume_rows if r["resume_id"] not in set(ordered_fb)]
        for rid in ordered_fb + remaining_fb:
            row = resume_rows_by_id.get(rid, {})
            uid = row.get("user_id", rid)
            if uid in seen_users_fb:
                continue
            seen_users_fb.add(uid)
            fallback_candidates.append(CandidateResult(
                user_id=uid,
                resume_id=rid,
                source_filename=row.get("source_filename", ""),
                similarity_score=None,
                name=row.get("name"),
                email=row.get("email"),
                phone=row.get("phone"),
                location=row.get("location"),
                work_experience_years=row.get("work_experience_years"),
                skills=list(row.get("skills") or []),
                objectives=row.get("objectives"),
                matched_sections=[],
                match_type="rds",
                match_reason="SQL filter matched — vector store empty, results ordered by SQL relevance",
                matched_chunk_text=row.get("work_experience_text") or row.get("projects") or row.get("objectives"),
            ))
            if len(fallback_candidates) == k:
                break
        logs = QueryLogs(
            user_query=request.query,
            sql_query=sql,
            sql_matched_count=sql_matched_count,
            routing_decision="rds_and_vector",
            routing_reason=routing_reason + " (vector store empty — showing SQL results)",
            vector_search_used=False,
            vector_query=request.query,
            timing_ms={**timing, "total_ms": _ms(t_total)},
            op_counts=op_counts,
        )
        return RetrieveResponse(query=request.query, k_used=k, candidates=fallback_candidates, logs=logs)

    # ── Aggregate best score + winning chunk per resume ────────────────
    best_score: dict[str, float] = {}
    best_chunk_id: dict[str, str] = {}
    best_section: dict[str, str] = {}
    matched_sections: dict[str, list[str]] = {}

    for cid, score in score_by_chunk.items():
        rid = chunk_to_resume[cid]
        section = chunk_to_section[cid]
        if rid not in best_score or score > best_score[rid]:
            best_score[rid] = score
            best_chunk_id[rid] = cid
            best_section[rid] = section
        matched_sections.setdefault(rid, [])
        if section not in matched_sections[rid]:
            matched_sections[rid].append(section)

    t0 = time.perf_counter()
    ranked_ids = sorted(best_score, key=lambda r: best_score[r], reverse=True)

    # Deduplicate by user_id
    seen_users_v: set[str] = set()
    deduped_ids: list[str] = []
    for rid in ranked_ids:
        uid = resume_rows_by_id.get(rid, {}).get("user_id") or rid
        if uid not in seen_users_v:
            seen_users_v.add(uid)
            deduped_ids.append(rid)
        if len(deduped_ids) == k:
            break
    timing["ranking_dedup_ms"] = _ms(t0)
    op_counts["resumes_before_dedup_count"] = len(ranked_ids)
    op_counts["candidates_returned_count"] = len(deduped_ids)

    candidates_v: list[CandidateResult] = []
    for rid in deduped_ids:
        row = resume_rows_by_id.get(rid, {})
        sec = best_section.get(rid, "")
        score = best_score[rid]
        winning_cid = best_chunk_id.get(rid)

        # Prefer the actual matched chunk text; fall back to full section text
        matched_text = (chunk_to_text.get(winning_cid) if winning_cid else None) or row.get(sec)

        candidates_v.append(CandidateResult(
            user_id=row.get("user_id", rid),
            resume_id=rid,
            source_filename=row.get("source_filename", ""),
            similarity_score=round(score, 6),
            name=row.get("name"),
            email=row.get("email"),
            phone=row.get("phone"),
            location=row.get("location"),
            work_experience_years=row.get("work_experience_years"),
            skills=list(row.get("skills") or []),
            objectives=row.get("objectives"),
            matched_sections=matched_sections.get(rid, []),
            match_type="vector",
            match_reason=f"Semantic vector search matched on '{sec.replace('_', ' ')}' section — cosine similarity {score:.1%}",
            matched_chunk_text=matched_text,
        ))

    sections_used = set(chunk_to_section[cid] for cid in score_by_chunk)
    vector_section_used = next(iter(sections_used)) if len(sections_used) == 1 else "mixed"

    logs = QueryLogs(
        user_query=request.query,
        sql_query=sql,
        sql_matched_count=sql_matched_count,
        routing_decision="rds_and_vector",
        routing_reason=routing_reason,
        vector_search_used=True,
        vector_section_used=vector_section_used,
        vector_query=hypothetical_doc or request.query,
        timing_ms={**timing, "total_ms": _ms(t_total)},
        op_counts=op_counts,
    )

    logger.info(
        "RDS + Vector complete: %d unique candidates from %d resumes | hyde=%s",
        len(candidates_v), len(resume_rows), embed_as_doc,
    )
    return RetrieveResponse(query=request.query, k_used=k, candidates=candidates_v, logs=logs)
