# Contributing

Thanks for taking a look. This is a portfolio project, but it's set up like a real
codebase, so the workflow below should feel familiar.

## Getting set up

Install [uv](https://docs.astral.sh/uv/); it manages the project's Python version
and dependencies, so uv is the only prerequisite.

```bash
uv sync
# create a .env file with:
#   OPENAI_API_KEY=...        (required)
#   GITHUB_TOKEN=...          (optional)
```

`GITHUB_TOKEN` only raises the GitHub raw-content rate limit during a fetch;
everything else runs without it.

## Project layout

```
src/hf_rag/
  ingestion/   Fetcher → Chunker → Embedder
  retrieval/   DenseRetriever, RerankingRetriever (both implement the Retriever Protocol)
  eval/        eval-set generation, metrics, comparison runner
  agent/       LangGraph agent, search_docs tool, query rewriter
  chat/        CLI / REPL entry point
notebooks/     corpus inspection and retrieval evaluation
tests/         unit tests
```

The one rule worth internalizing: anything that consumes retrieval depends on the
`Retriever` Protocol, never on a concrete retriever. A new retrieval strategy should
implement the Protocol so the agent and the eval harness pick it up unchanged.

## Running the pipeline

```bash
uv run hf-fetch    # pin upstream docs at the current main SHA and download
uv run hf-chunk    # heading-aware, token-budgeted chunking
uv run hf-embed    # embed into ChromaDB
uv run hf-eval     # dense-vs-rerank comparison with bootstrap CIs
uv run hf-chat     # interactive REPL
```

Switch the embedding backend with `EMBEDDING_CONFIG` (`text-embedding-3-small` or
`all-MiniLM-L6-v2`); each writes to its own ChromaDB collection, so both can coexist.

## Tests

```bash
uv run pytest
```

Tests are unit-level only and must not call an LLM or hit the network. Anything that
depends on live model behavior belongs in a notebook, not the test suite, so the
suite stays fast and deterministic.

## Code conventions

- **Stateful components are classes** — `Fetcher`, `Chunker`, and `Embedder` hold
  configuration and state; pure transforms can stay functions.
- **Data objects are `@dataclass`es** — use them for DTOs like `Chunk` and
  `EvalItem` rather than passing loose dicts around.
- **Type hints everywhere**, including return types. Use `Optional[str]`, not
  `str | None`, to stay consistent with the existing code.
- **`pathlib.Path` for filesystem paths**, never `os.path`.
- **Logging via `loguru`**, not the stdlib `logging` module.

## Regenerating the corpus and eval set

The chunk corpus and the frozen eval set are intentionally stable. `hf-eval` skips
regeneration when `data/eval_set.json` already exists, so the dense-vs-rerank numbers
stay comparable across runs. `data/corpus_manifest.json`, `data/eval_set.json`, and
`data/eval_results.json` are tracked in git for this reason; the rest of `data/`
(raw pages, chunks, the ChromaDB collections) is regenerable and stays ignored.

They must be regenerated **together**. Any change to chunking that alters chunk text
or IDs makes the eval set's chunk-ID references stale. When that happens, delete
`data/eval_set.json` and the affected ChromaDB collection, then re-run
`hf-chunk → hf-embed → hf-eval`. Regenerating the corpus without regenerating the
eval set silently invalidates the comparison. If you do regenerate it, commit the
updated `data/eval_set.json` and `data/eval_results.json` so the README numbers and
the repo stay in sync.

## Submitting changes

- Keep changes scoped: one concern per PR.
- A pre-commit hook (`ruff check`, `ruff format`, `pytest`) runs on every commit and
  blocks it on failure — `uv run pre-commit install` once after cloning to enable it.
- If a change affects retrieval quality, include the relevant `hf-eval` numbers
  (with confidence intervals) in the PR description, the same way the README reports
  them. Evidence over assertion is the point of this project.
