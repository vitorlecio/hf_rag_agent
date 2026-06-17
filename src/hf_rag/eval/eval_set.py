import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from hf_rag.config import EVAL_SET_PATH
from hf_rag.ingestion.chunker import Chunk

_GENERATE_MODEL = "gpt-4.1-mini"
_PROMPT = """\
You are building a retrieval evaluation set for a RAG system about HuggingFace Transformers.
Given the documentation chunk below, write one natural-language question that:
- A practitioner would realistically ask
- Is answerable using ONLY the content of this chunk
- Is specific enough that this exact chunk is the best source

Respond with just the question, nothing else.

Chunk:
{content}"""


@dataclass
class EvalItem:
    query: str
    relevant_chunk_ids: list[str]


def load(path: Path = EVAL_SET_PATH) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        return [EvalItem(**item) for item in json.load(f)]


def save(items: list[EvalItem], path: Path = EVAL_SET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in items], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(items)} eval items to {path}")


def generate(chunks: list[Chunk], path: Path = EVAL_SET_PATH) -> list[EvalItem]:
    """Generate one question per chunk using GPT. Skips if eval set already exists."""
    if path.exists():
        logger.info(
            f"Eval set already exists at {path} — loading instead of regenerating"
        )
        return load(path)

    import openai

    client = openai.OpenAI()

    items: list[EvalItem] = []
    for i, chunk in enumerate(chunks):
        try:
            response = client.chat.completions.create(
                model=_GENERATE_MODEL,
                messages=[
                    {"role": "user", "content": _PROMPT.format(content=chunk.content)}
                ],
                temperature=0.3,
            )
            query = response.choices[0].message.content.strip()
            items.append(EvalItem(query=query, relevant_chunk_ids=[chunk.chunk_id]))
            logger.debug(f"[{i + 1}/{len(chunks)}] {query[:80]}")
        except Exception as exc:
            logger.warning(f"Skipped {chunk.chunk_id}: {exc}")

    save(items, path)
    return items
