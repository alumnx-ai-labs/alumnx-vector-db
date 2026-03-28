import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

from app.routers.documents import router as documents_router
from app.routers.ingest import router as ingest_router
from app.routers.retrieve import router as retrieve_router
from app.errors import error_response
from app.services.store.postgres_store import PostgresStore


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("nexvec.main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Running DB migrations...")
    PostgresStore().ensure_table()
    logger.info("DB ready.")
    yield


app = FastAPI(title="NexVec", version="1.4.0", lifespan=lifespan)

app.include_router(documents_router)
app.include_router(ingest_router)
app.include_router(retrieve_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "NexVec API is running! Access the docs at /docs"}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail and "message" in detail:
        return error_response(exc.status_code, detail["error"], detail["message"], detail.get("detail"))
    if isinstance(detail, dict):
        return error_response(exc.status_code, detail.get("error", "HTTP_ERROR"), detail.get("message", str(detail)), detail.get("detail"))
    return error_response(exc.status_code, "HTTP_ERROR", str(detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "VALIDATION_ERROR", "Validation error.", {"errors": exc.errors()})
