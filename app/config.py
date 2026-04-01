from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    knn_k: int
    embedding_model: str
    output_dimensionality: int
    vector_size: int
    vector_store_path: Path
    min_page_text_length: int
    postgres_url: str
    s3_bucket_name: str
    s3_region: str
    s3_access_key: str
    s3_secret_key: str


_CONFIG_CACHE: AppConfig | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw_config() -> dict[str, Any]:
    config_path = project_root() / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_config() -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    raw = _load_raw_config()
    vector_store_path = Path(raw.get("vector_store_path", "./vector_store/"))
    if not vector_store_path.is_absolute():
        vector_store_path = (project_root() / vector_store_path).resolve()

    postgres_url = os.environ.get("POSTGRES_URL", "")
    if not postgres_url:
        db_user = os.environ.get("DB_USER", "")
        db_pass = os.environ.get("DB_PASSWORD", "")
        db_host = os.environ.get("DB_HOST", "")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "")
        if db_user and db_host and db_name:
            postgres_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        else:
            raise RuntimeError("POSTGRES_URL environment variable is not set, nor are individual DB_* variables.")

    _CONFIG_CACHE = AppConfig(
        knn_k=int(raw["knn_k"]),
        embedding_model=str(raw["embedding_model"]),
        output_dimensionality=int(raw["output_dimensionality"]),
        vector_size=int(raw["vector_size"]),
        vector_store_path=vector_store_path,
        min_page_text_length=int(raw["min_page_text_length"]),
        postgres_url=postgres_url,
        s3_bucket_name=os.environ.get("BUCKET_NAME", "").strip(' "'),
        s3_region=os.environ.get("Region", "ap-south-1").strip(' "'),
        s3_access_key=os.environ.get("S3_Access_Key_Id", "").strip(' "'),
        s3_secret_key=os.environ.get("S3_Access_Secret", "").strip(' "'),
    )
    return _CONFIG_CACHE
