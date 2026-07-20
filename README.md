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

**Phase 1, Module 1 (project scaffold)** — complete.

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

Ingestion, indexing, retrieval, generation, serving, and evaluation
instructions will be added here as their modules are built.
