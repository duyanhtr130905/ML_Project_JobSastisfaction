"""
Configuration settings for the translation system
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# MarianMT Configuration
MARIAN_MODEL = os.getenv("MARIAN_MODEL", "Helsinki-NLP/opus-mt-en-vi")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

# Translation Memory Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "translation_memory"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "en_vi_translations")

# Knowledge Graph Configuration
KG_PERSIST_DIR = os.getenv("KG_PERSIST_DIR", str(BASE_DIR / "data" / "knowledge_graph"))
KG_DEFAULT_DOMAIN = os.getenv("KG_DEFAULT_DOMAIN", "general_scientific")

# Database Configuration (for analytics)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "translation_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Superset Configuration
SUPERSET_HOST = os.getenv("SUPERSET_HOST", "localhost")
SUPERSET_PORT = int(os.getenv("SUPERSET_PORT", 8088))
SUPERSET_URL = f"http://{SUPERSET_HOST}:{SUPERSET_PORT}"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "translation.log"))

# Cache Configuration
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # seconds

# Performance Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.9))

# Security Configuration
API_KEY = os.getenv("API_KEY", None)  # Set in production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Feature Flags
ENABLE_TRANSLATION_MEMORY = os.getenv("ENABLE_TRANSLATION_MEMORY", "true").lower() == "true"
ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
ENABLE_ANALYTICS = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"

# Create data directories if they don't exist
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(KG_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
