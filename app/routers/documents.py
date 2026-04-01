from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.ingestion import UNIVERSAL_VECTOR_STORE
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.services.store.s3_store import S3Store


router = APIRouter()


@router.get("/documents/{filename}")
def get_document(filename: str) -> dict:
    pg = PostgresStore()
    doc = pg.get_document(filename)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail={"error": "DOCUMENT_NOT_FOUND", "message": f"No active document found with filename '{filename}'."},
        )
    return doc


@router.get("/documents")
def list_documents() -> dict:
    pg = PostgresStore()
    return {"documents": pg.list_documents()}


@router.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    pg = PostgresStore()
    vfs = VectorFileStore()

    section_chunks = pg.delete_document(filename)
    if not section_chunks:
        raise HTTPException(
            status_code=404,
            detail={"error": "DOCUMENT_NOT_FOUND", "message": f"No active document found with filename '{filename}'."},
        )

    all_chunk_ids = {chunk_id for _, chunk_id in section_chunks}
    if all_chunk_ids:
        vfs.remove_chunk_ids(UNIVERSAL_VECTOR_STORE, all_chunk_ids)

    s3_store = S3Store()
    s3_store.delete_file(filename)

    return {"deleted": filename, "chunks_removed": len(section_chunks)}
