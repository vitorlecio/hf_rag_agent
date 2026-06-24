import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger
from transformers import AutoTokenizer

from hf_rag.ingestion.fetcher import RawPage

TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200  # tokens, counted with TOKENIZER_MODEL
CHUNK_OVERLAP = 0  # zero overlap — each chunk is a clean eval unit

_HEADERS_TO_SPLIT = [("##", "h2"), ("###", "h3")]


@dataclass
class Chunk:
    chunk_id: str  # deterministic, e.g. "training_001"
    page_path: str  # e.g. "training.md"
    page_title: str  # e.g. "Training"
    heading: str  # most specific heading this chunk falls under
    content: str  # "# {page_title}\n## {heading}\n\n{text}" — prefix included, heading omitted if none
    token_count: int  # verified; logged if > CHUNK_SIZE (atomic blocks may exceed)


class Chunker:
    def __init__(self, tokenizer_model: str = TOKENIZER_MODEL) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT,
            strip_headers=False,  # keep ## / ### lines in content for retrieval
        )
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def chunk_pages(self, pages: list[RawPage]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for page in pages:
            page_chunks = self._chunk_page(page)
            all_chunks.extend(page_chunks)
            logger.debug(f"{page.path}: {len(page_chunks)} chunks")
        logger.info(f"Total: {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks

    def _chunk_page(self, page: RawPage) -> list[Chunk]:
        content = re.sub(r"^#\s+[^\n]+\n?", "", page.content).strip()
        sections = self._md_splitter.split_text(content)
        chunks: list[Chunk] = []
        idx = 0
        for section in sections:
            heading = section.metadata.get("h3") or section.metadata.get("h2") or ""
            # Header line is dropped here and re-added below via `prefix`, so every split
            # piece carries the heading anchor, not just the first piece of the section.
            text = re.sub(
                r"^#{2,3}\s+[^\n]+\n*", "", section.page_content.strip()
            ).strip()
            if not text:
                continue
            prefix = f"# {page.title}" + (f"\n## {heading}" if heading else "")
            for piece in self._split_section(text):
                content = f"{prefix}\n\n{piece}"
                token_count = len(self._tokenizer.encode(content))
                if token_count > CHUNK_SIZE:
                    logger.debug(
                        f"Over budget ({token_count} tokens): "
                        f"{page.path} / {heading or 'top-level'}"
                    )
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(page.path, idx),
                        page_path=page.path,
                        page_title=page.title,
                        heading=heading,
                        content=content,
                        token_count=token_count,
                    )
                )
                idx += 1
        return chunks

    def _split_section(self, text: str) -> list[str]:
        """Tables kept whole; code-mixed sections split at fences; pure prose token-split."""
        if re.search(r"^\|", text, re.MULTILINE):
            return [text]
        if "```" not in text:
            return self._text_splitter.split_text(text) or [text]
        return self._split_around_fences(text)

    def _split_around_fences(self, text: str) -> list[str]:
        """Split at code fence boundaries, then greedily merge atoms up to CHUNK_SIZE."""
        raw = re.split(r"(```.*?```)", text, flags=re.DOTALL)

        atoms: list[str] = []
        for seg in raw:
            seg = seg.strip()
            if not seg:
                continue
            if seg.startswith("```"):
                atoms.append(seg)
            else:
                sub = self._text_splitter.split_text(seg)
                atoms.extend(sub if sub else [seg])

        chunks: list[str] = []
        current = ""
        for atom in atoms:
            candidate = (current + "\n\n" + atom).strip() if current else atom
            if len(self._tokenizer.encode(candidate)) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = atom
        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    @staticmethod
    def load(path: Path = Path("data/raw_pages.json")) -> list[RawPage]:
        with open(path, encoding="utf-8") as f:
            return [RawPage(**p) for p in json.load(f)]

    @staticmethod
    def save(chunks: list[Chunk], path: Path = Path("data/chunks.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {path}")


def _make_chunk_id(page_path: str, idx: int) -> str:
    base = page_path.replace("/", "_").replace(".md", "")
    return f"{base}_{idx:03d}"


def main() -> None:
    pages = Chunker.load()
    chunker = Chunker()
    chunks = chunker.chunk_pages(pages)
    Chunker.save(chunks)
