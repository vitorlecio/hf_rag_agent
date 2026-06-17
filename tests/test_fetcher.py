from unittest.mock import MagicMock, patch

import pytest
import requests

from hf_rag.ingestion.fetcher import Fetcher, RawPage, _parse_page


# ---------------------------------------------------------------------------
# _parse_page (pure function)
# ---------------------------------------------------------------------------


class TestParsePage:
    def test_extracts_title_from_frontmatter(self) -> None:
        content = '---\ntitle: "Fine-Tuning Guide"\n---\n\nSome content.'
        title, body = _parse_page(content, "training.md")
        assert title == "Fine-Tuning Guide"
        assert "---" not in body

    def test_strips_yaml_frontmatter_from_body(self) -> None:
        content = "---\ntitle: Guide\n---\n\nReal content here."
        _, body = _parse_page(content, "training.md")
        assert body == "Real content here."

    def test_falls_back_to_h1_heading(self) -> None:
        content = "# Trainer Overview\n\nSome text."
        title, _ = _parse_page(content, "trainer.md")
        assert title == "Trainer Overview"

    def test_falls_back_to_filename(self) -> None:
        content = "No frontmatter, no heading."
        title, _ = _parse_page(content, "perf_train_gpu_one.md")
        assert title == "Perf Train Gpu One"

    def test_strips_html_comments(self) -> None:
        content = "<!-- Copyright 2023 HuggingFace -->\n\n# Title\n\nContent."
        _, body = _parse_page(content, "training.md")
        assert "<!--" not in body
        assert "Copyright" not in body

    def test_strips_single_line_html_comment(self) -> None:
        content = "<!-- license: Apache -->\n# My Page\n\nBody."
        _, body = _parse_page(content, "x.md")
        assert "<!--" not in body

    def test_quoted_title_strips_quotes(self) -> None:
        content = "---\ntitle: 'The PEFT Guide'\n---\n\nContent."
        title, _ = _parse_page(content, "peft.md")
        assert title == "The PEFT Guide"
        assert "'" not in title


# ---------------------------------------------------------------------------
# Fetcher._fetch_file (mocked HTTP)
# ---------------------------------------------------------------------------

SAMPLE_CONTENT = "---\ntitle: Training\n---\n\n# Training\n\nHere is the guide."


@pytest.fixture
def fetcher() -> Fetcher:
    return Fetcher(token=None)


class TestFetchFile:
    def test_returns_raw_page_on_success(self, fetcher: Fetcher) -> None:
        response = MagicMock()
        response.status_code = 200
        response.text = SAMPLE_CONTENT
        response.raise_for_status = MagicMock()

        with patch.object(fetcher._session, "get", return_value=response):
            result = fetcher._fetch_file("training.md", sha="abc123")

        assert isinstance(result, RawPage)
        assert result.path == "training.md"
        assert result.title == "Training"

    def test_returns_none_on_404(self, fetcher: Fetcher) -> None:
        response = MagicMock()
        response.status_code = 404

        with patch.object(fetcher._session, "get", return_value=response):
            result = fetcher._fetch_file("missing.md", sha="abc123")

        assert result is None

    def test_returns_none_on_request_exception(self, fetcher: Fetcher) -> None:
        with patch.object(
            fetcher._session, "get", side_effect=requests.RequestException("timeout")
        ):
            result = fetcher._fetch_file("training.md", sha="abc123")

        assert result is None

    def test_source_url_contains_sha_and_path(self, fetcher: Fetcher) -> None:
        response = MagicMock()
        response.status_code = 200
        response.text = SAMPLE_CONTENT
        response.raise_for_status = MagicMock()

        with patch.object(fetcher._session, "get", return_value=response):
            result = fetcher._fetch_file("training.md", sha="deadbeef")

        assert result is not None
        assert "deadbeef" in result.source_url
        assert "training.md" in result.source_url


# ---------------------------------------------------------------------------
# Fetcher.save / write_manifest
# ---------------------------------------------------------------------------


class TestStaticWriters:
    def test_save_roundtrip(self, tmp_path, fetcher: Fetcher) -> None:
        pages = [
            RawPage(
                path="training.md",
                title="Training",
                content="Content.",
                source_url="https://raw.githubusercontent.com/x",
            )
        ]
        dest = tmp_path / "raw_pages.json"
        Fetcher.save(pages, path=dest)
        assert dest.exists()

        from hf_rag.ingestion.chunker import Chunker

        loaded = Chunker.load(dest)
        assert len(loaded) == 1
        assert loaded[0].path == "training.md"

    def test_write_manifest_creates_file(self, tmp_path, fetcher: Fetcher) -> None:
        pages = [
            RawPage(
                path="trainer.md",
                title="Trainer",
                content="Content.",
                source_url="https://raw.githubusercontent.com/x",
            )
        ]
        dest = tmp_path / "manifest.json"
        Fetcher.write_manifest(pages, sha="abc" * 13, path=dest)
        assert dest.exists()

        import json

        with open(dest) as f:
            manifest = json.load(f)
        assert manifest["count"] == 1
        assert "trainer.md" in manifest["files"]
