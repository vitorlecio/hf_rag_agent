from pathlib import Path
from typing import Callable

# Type alias shared by Embedder and Retriever
EmbeddingFunction = Callable[[list[str]], list[list[float]]]

# Filesystem paths
CHROMA_DIR = Path("data/chroma")
CHUNKS_PATH = Path("data/chunks.json")
RAW_PAGES_PATH = Path("data/raw_pages.json")
MANIFEST_PATH = Path("data/corpus_manifest.json")
EVAL_SET_PATH = Path("data/eval_set.json")

# ChromaDB collection names — one per embedding config so both can coexist
COLLECTION_OPENAI = "hf_docs_openai"
COLLECTION_MINILM = "hf_docs_minilm"

# Retrieval parameters
DENSE_TOP_N = 20  # candidates fetched from ChromaDB before reranking
RERANK_TOP_K = 5  # chunks returned to the agent after reranking
DENSE_TOP_K = 5  # chunks returned to the agent in dense-only mode

# Model identifiers
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
MINILM_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "gpt-4.1-mini"


def make_openai_embedding_fn() -> EmbeddingFunction:
    """Embedding function backed by text-embedding-3-small (OpenAI API)."""
    import openai

    client = openai.OpenAI()

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    return embed


def make_minilm_embedding_fn() -> EmbeddingFunction:
    """Embedding function backed by all-MiniLM-L6-v2 (local, no API cost)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MINILM_EMBEDDING_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, convert_to_numpy=True).tolist()

    return embed
