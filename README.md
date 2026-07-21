# Financial RAG

An analyst-primary Retrieval-Augmented Generation (RAG) system over financial
documents. A research analyst asks a plain-language question and gets a
grounded, cited answer drawn only from the source documents — with an honest
refusal when supporting context is weak.

Corpus: SEC filings (10-K, 10-Q), earnings-call transcripts, and synthetic
investment memos, for three peer companies (KO, PEP, MDLZ) across two
quarters.

See [CLAUDE.md](./CLAUDE.md) for the full project spec, architecture, and
build plan. This project is being built module by module, in order; install
and usage instructions will grow here as each module lands.

## Status

**Phase 1, Module 2 (ingestion)** — complete.
**Phase 1, Module 3 (indexing)** — complete.
**Phase 1, Module 4 (retrieval)** — complete.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and adjust any values you want to override
(all settings have working defaults):

```bash
cp .env.example .env
```

## Running

### Ingestion

Source PDFs live under `data/<TICKER>/`, named `<TICKER>_<DOCTYPE>_<PERIOD>.pdf`
(e.g. `KO_10-K_FY2025.pdf`, `KO_transcript_2026Q1.pdf`). Run ingestion with:

```bash
uv run python -m src.ingestion.pipeline
```

This reads every PDF under `data/`, tags each chunk with metadata parsed from
its filename, splits filings by item and transcripts by speaker turn, and
writes the full chunk list to `data/processed/chunks.jsonl` (JSON Lines, one
chunk per line), overwriting it each run. Re-running is safe — it always
rebuilds from scratch, so it never duplicates chunks.

Indexing, retrieval, generation, serving, and evaluation instructions will be
added here as their modules are built.
### Indexing

Indexing consumes `data/processed/chunks.jsonl` (the metadata-rich chunks
produced by ingestion) and builds both search indexes from them in one pass:

```bash
uv run python -m src.indexing.pipeline
```

This embeds every chunk with bge-base-en-v1.5 into a persistent Chroma
vector store at `data/chroma/`, builds a BM25 keyword index over the same
chunks and persists it to `data/bm25/bm25.pkl`, and runs one sample
similarity query so you can eyeball the results. Re-running always rebuilds
both indexes from scratch, so it never leaves duplicate or stale vectors
behind.

### Retrieval

Retrieval consumes Module 3's persisted Chroma and BM25 indexes directly --
nothing is rebuilt or re-embedded. It runs three stages: hybrid retrieval
(dense + BM25, merged by reciprocal rank fusion) pulls a wide candidate
pool, an optional metadata filter scopes it by company and/or period, then
the bge-reranker-base cross-encoder re-scores what's left for final
precision.

```bash
uv run python -m src.retrieval.pipeline "What drove revenue growth this quarter?"
uv run python -m src.retrieval.pipeline "What drove revenue growth this quarter?" --company KO
uv run python -m src.retrieval.pipeline "What happened this quarter?" --period 2025Q3
```

Prints all three stages (hybrid candidates, post-filter, post-rerank) so you
can see exactly what each stage changed. Every result carries its full chunk
metadata (company, doc_type, period, section_or_speaker, source_path,
chunk_id), so its source is always traceable.

Generation, serving, and evaluation instructions will be added here as their
modules are built.
