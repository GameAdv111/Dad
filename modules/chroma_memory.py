import os
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.path.join("memory", "chroma")
os.makedirs(CHROMA_DIR, exist_ok=True)

_LOCAL_EMBED_PATH = os.path.join("models", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(_LOCAL_EMBED_PATH, local_files_only=True)

_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name="jarvis_memory",
    metadata={"hnsw:space": "cosine"},
)

def _embed(text: str):
    return _embedder.encode(
        text,
        normalize_embeddings=True
    ).tolist()

def save_memory(role: str, content: str):

    ts = datetime.now().isoformat(timespec="seconds")
    doc = f"{role}: {content}"

    uid = f"{ts}-{abs(hash(doc))}"

    try:
        _collection.add(
            ids=[uid],
            documents=[doc],
            metadatas=[{"role": role, "timestamp": ts}],
            embeddings=[_embed(doc)],
        )
    except Exception as e:
        print("Memory add error:", e)

def search_memory(query: str, limit: int = 6):

    if _collection.count() == 0:
        return []

    try:
        res = _collection.query(
            query_embeddings=[_embed(query)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print("Memory search error:", e)
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []

    for doc, meta, dist in zip(docs, metas, dists):
        if dist is not None and dist > 0.45:
            continue

        ts = (meta or {}).get("timestamp", "")
        out.append(f"[{ts}] {doc}")

    return out