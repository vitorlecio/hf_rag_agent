from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    page_path: str
    page_title: str
    heading: str


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...
