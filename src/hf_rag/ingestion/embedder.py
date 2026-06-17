import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from loguru import logger

from hf_rag.config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_MINILM,
    COLLECTION_OPENAI,
    EmbeddingFunction,
    make_minilm_embedding_fn,
    make_openai_embedding_fn,
)
from hf_rag.ingestion.chunker import Chunk

EMBED_BATCH_SIZE = 100


class Embedder:
    def __init__(
        self,
        collection_name: str,
        embedding_fn: EmbeddingFunction,
        persist_dir: Path = CHROMA_DIR,
    ) -> None:
        self._embedding_fn = embedding_fn
        client = chromadb.PersistentClient(path=str(persist_dir))
        # No embedding_function passed to ChromaDB — embeddings are pre-computed.
        # This keeps ChromaDB agnostic to the model and prevents silent default drift.
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{collection_name}' ready at {persist_dir}")

    def embed(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            self._upsert_batch(batch)
            logger.debug(
                f"Upserted {min(i + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}"
            )
        logger.info(f"Embedded {len(chunks)} chunks into '{self._collection.name}'")

    def _upsert_batch(self, chunks: list[Chunk]) -> None:
        texts = [c.content for c in chunks]
        embeddings = self._embedding_fn(texts)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "page_path": c.page_path,
                    "page_title": c.page_title,
                    "heading": c.heading,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )

    @staticmethod
    def load(path: Path = CHUNKS_PATH) -> list[Chunk]:
        with open(path, encoding="utf-8") as f:
            return [Chunk(**c) for c in json.load(f)]


def main() -> None:
    load_dotenv()

    config = os.getenv("EMBEDDING_CONFIG", "openai").lower()
    if config == "openai":
        collection_name = COLLECTION_OPENAI
        embedding_fn = make_openai_embedding_fn()
    elif config == "minilm":
        collection_name = COLLECTION_MINILM
        embedding_fn = make_minilm_embedding_fn()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_CONFIG={config!r}. Use 'openai' or 'minilm'."
        )

    chunks = Embedder.load()
    embedder = Embedder(collection_name=collection_name, embedding_fn=embedding_fn)
    embedder.embed(chunks)
