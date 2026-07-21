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

**Phase 1, Module 3 (indexing)** — complete.

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

Retrieval, generation, serving, and evaluation instructions will be added
here as their modules are built.
