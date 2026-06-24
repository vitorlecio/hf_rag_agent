import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from loguru import logger

REPO = "huggingface/transformers"
DOCS_PREFIX = "docs/source/en"
DEFAULT_BRANCH = "main"
REQUEST_DELAY = 0.5  # seconds between requests

# Paths relative to DOCS_PREFIX. Missing files are logged and skipped.
FILES: list[str] = [
    # Core training workflow — primary multi-hop chain
    "training.md",
    "trainer.md",
    "peft.md",
    # Quantization guides
    "quantization/overview.md",
    "quantization/bitsandbytes.md",
    "quantization/gptq.md",
    # NLP task guides
    "tasks/sequence_classification.md",
    "tasks/token_classification.md",
    "tasks/question_answering.md",
    "tasks/summarization.md",
    "tasks/translation.md",
    "tasks/multiple_choice.md",
    "tasks/masked_language_modeling.md",
    # Vision task guides
    "tasks/image_classification.md",
    "tasks/object_detection.md",
    "tasks/visual_question_answering.md",
]


@dataclass
class RawPage:
    path: str  # relative to DOCS_PREFIX, e.g. "tasks/sequence_classification.md"
    title: str  # extracted from frontmatter or first heading
    content: str  # markdown with frontmatter and copyright comments stripped
    source_url: str  # full GitHub raw URL


class Fetcher:
    def __init__(
        self,
        repo: str = REPO,
        branch: str = DEFAULT_BRANCH,
        token: Optional[str] = None,
    ) -> None:
        self._repo = repo
        self._branch = branch
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "hf-rag-agent/0.1"
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    def resolve_head_sha(self) -> str:
        url = f"https://api.github.com/repos/{self._repo}/branches/{self._branch}"
        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["commit"]["sha"]

    def fetch_all(self, sha: str) -> list[RawPage]:
        pages: list[RawPage] = []
        for path in FILES:
            page = self._fetch_file(path, sha)
            if page is not None:
                pages.append(page)
            time.sleep(REQUEST_DELAY)
        logger.info(f"Fetched {len(pages)} / {len(FILES)} pages")
        return pages

    def _fetch_file(self, path: str, sha: str) -> Optional[RawPage]:
        url = (
            f"https://raw.githubusercontent.com/{self._repo}/{sha}/{DOCS_PREFIX}/{path}"
        )
        try:
            response = self._session.get(url, timeout=10)
            if response.status_code == 404:
                logger.warning(f"Not found (skipping): {path}")
                return None
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"Failed to fetch {path}: {exc}")
            return None

        title, content = _parse_page(response.text, path)
        return RawPage(path=path, title=title, content=content, source_url=url)

    @staticmethod
    def save(pages: list[RawPage], path: Path = Path("data/raw_pages.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in pages], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(pages)} pages to {path}")

    @staticmethod
    def write_manifest(
        pages: list[RawPage],
        sha: str,
        repo: str = REPO,
        path: Path = Path("data/corpus_manifest.json"),
    ) -> None:
        manifest = {
            "repo": repo,
            "commit_sha": sha,
            "files": [p.path for p in pages],
            "count": len(pages),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {path} ({len(pages)} files, sha={sha[:8]})")


def _parse_page(content: str, path: str) -> tuple[str, str]:
    """Return (title, clean_content) with copyright comments and frontmatter stripped."""
    # Strip HTML comment blocks (copyright headers common in HF docs)
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL).strip()

    title = ""

    # Parse YAML frontmatter if present
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end]
            content = content[end + 3 :].strip()
            match = re.search(r"^title:\s*(.+)$", frontmatter, re.MULTILINE)
            if match:
                title = match.group(1).strip().strip("\"'")

    # Fall back to first # heading
    if not title:
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            title = match.group(1).strip()

    # Final fallback: derive from filename
    if not title:
        title = Path(path).stem.replace("_", " ").title()

    return title, content


def main() -> None:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")

    fetcher = Fetcher(token=token)

    logger.info(f"Resolving HEAD sha for {REPO}/{DEFAULT_BRANCH}...")
    sha = fetcher.resolve_head_sha()
    logger.info(f"Pinned to sha: {sha}")

    pages = fetcher.fetch_all(sha)
    Fetcher.save(pages)
    Fetcher.write_manifest(pages, sha)
