import os
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Папка для базы
CHROMA_DIR = os.path.join("memory", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Локальная embedding-модель (быстрая и лёгкая)
_LOCAL_EMBED_PATH = os.path.join("models", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(_LOCAL_EMBED_PATH, local_files_only=True)

# Chroma persistent client (локально на диске)
_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name="jarvis_memory",
    metadata={"hnsw:space": "cosine"},
)

def _embed(text: str) -> list[float]:
    # Chroma ждёт list[float]
    vec = _embedder.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()

def save_memory(role: str, content: str, metadata: dict | None = None) -> None:
    """
    Сохраняет сообщение в "неограниченную" память.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    doc = f"{role}: {content}"
    meta = {"role": role, "timestamp": ts}
    if metadata:
        meta.update(metadata)

    # Уникальный id: timestamp + hash контента
    uid = f"{ts}-{abs(hash(doc))}"

    _collection.add(
        ids=[uid],
        documents=[doc],
        metadatas=[meta],
        embeddings=[_embed(doc)],
    )

def search_memory(query: str, limit: int = 6) -> list[str]:
    """
    Возвращает limit самых релевантных "воспоминаний".
    """
    if limit <= 0:
        return []

    res = _collection.query(
        query_embeddings=[_embed(query)],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[str] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # можно фильтровать "слишком далёкие" совпадения
        # cosine distance: 0 = идеально, больше = хуже
        if dist is not None and dist > 0.45:
            continue
        ts = (meta or {}).get("timestamp", "")
        out.append(f"[{ts}] {doc}")

    return out