# CLAUDE.md

Project instructions for the AI coding agent. Read this file first, every session, before writing any code.

---

## 0. Rules of engagement (read before doing anything)

1. **Build only within the current active phase.** The active phase is stated in Section 6. Do not write code for a later phase, and do not import or scaffold later-phase tools, unless the human explicitly says to.
2. **One module at a time.** Build a single module, then stop. Do not chain modules together in one pass.
3. **Stop for review after each module.** After finishing a module, summarize what you built, how to run it, and how to verify it. Then wait. The human will test, review the diff, and commit before you continue.
4. **Ask before crossing a phase boundary.** When the current phase's exit checklist (Section 7) is met, do not start the next phase on your own. Ask first.
5. **Prefer clarity over cleverness.** This is a learning project and a portfolio piece. Readable, well-commented code beats compact code.
6. **When something is ambiguous, ask one question rather than guessing.**

---

## 1. What this project is

An analyst-primary Retrieval-Augmented Generation (RAG) system over financial documents. A research analyst asks a plain-language question and gets a grounded, cited answer drawn only from the source documents. The system refuses to answer when its supporting context is weak, rather than guessing.

- **Primary user:** research analyst at a mid-size asset manager.
- **Secondary user (Phase 2):** compliance officer, served by the same auditability guarantees.
- **Corpus:** SEC filings (10-K, 10-Q), earnings-call transcripts, and synthetic investment memos, for three peer companies (KO, PEP, MDLZ) across two quarters.

**Non-goals (never build these):** investment or compliance recommendations, trade execution, live market data, user accounts or multi-tenancy.

---

## 2. Hard rules (global, every phase)

- **Open-source only.** Cost-effectiveness is a core constraint. No paid services in the default path.
- **LangChain** is the orchestration framework.
- **PDF-only corpus.** All source documents are PDFs. Use PyMuPDF. Do not add HTML or web loaders.
- **Local-first and container-portable.** It must run on a laptop and containerize cleanly.
- **Design for compliance, ship for the analyst.** Three things are non-negotiable from day one because Phase 2 tightens them rather than adds them:
  - Rich metadata on every chunk (company, document type, quarter, section or speaker).
  - Citation is a real component, not an afterthought.
  - The refusal gate is a real component with a tunable confidence threshold.
- **Grounding over fluency.** A shorter, fully sourced answer beats a fuller answer that drifts from the documents.
- **Reproducible ingestion.** Re-running ingestion must not duplicate content.

---

## 3. Full target architecture (REFERENCE ONLY — the end state)

This is the north star, included so that Phase 1 decisions are made knowing what plugs in later. **Do not build ahead of the active phase.**

- **Phase 1 — Core RAG:** Ingestion, Indexing, Retrieval, Generation, Serving, plus cross-cutting Evaluation and Observability.
- **Phase 2 — Structured data:** DuckDB (analytical store) and dbt (models and quality tests). Adds structured and entity extraction to ingestion. dbt tests gate data quality.
- **Phase 3 — Graph retrieval:** Neo4j knowledge graph and a GraphRAG retrieval path running alongside hybrid search.
- **Phase 4 — MLOps:** MLflow (experiment tracking) and Airflow (orchestration of re-ingestion and re-evaluation), plus a CI/CD evaluation gate that blocks a merge when faithfulness drops.

The request flows left to right: Data Sources -> Ingestion -> Indexing -> Retrieval -> Generation -> Serving -> Analyst. Evaluation and Observability run alongside all stages.

---

## 4. Finalized stack, by layer

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| PDF loading | PyMuPDF |
| Chunking | Section-aware splitter (filing items, speaker turns) |
| Embeddings | bge-base-en-v1.5 (local, via sentence-transformers) |
| Vector store | Chroma (persistent, on disk) |
| Keyword index | BM25 (rank_bm25 or LangChain BM25Retriever) |
| Hybrid retrieval | LangChain EnsembleRetriever (dense + sparse) |
| Reranker | bge-reranker-base (cross-encoder, local) |
| LLM | Ollama (open-source model), behind a swappable interface |
| Citation | LangChain source-document tracking, enforced in the prompt |
| Refusal gate | Custom, threshold on reranker/retrieval score |
| Serving | FastAPI + Uvicorn, Docker |
| Demo UI (optional) | Streamlit |
| Evaluation | RAGAS |
| Observability | Structured logging (SQLite/JSON to start) |
| Config + deps | pydantic-settings + uv |

Later phases: DuckDB + dbt (P2), Neo4j (P3), MLflow + Airflow (P4).

---

## 5. Target repo structure

```
financial-rag/
  CLAUDE.md                # this file
  README.md
  pyproject.toml           # deps via uv
  .env.example
  config/
    settings.py            # pydantic-settings
  data/
    KO/  PEP/  MDLZ/        # source PDFs, named co_type_period
  src/
    ingestion/             # module 2
    indexing/              # module 3
    retrieval/             # module 4
    generation/            # module 5
    serving/               # module 6
    evaluation/            # module 7
    common/                # shared models, metadata schema
  eval/
    question_set.xlsx      # the evaluation answer key
  tests/
  docker/
    Dockerfile
    docker-compose.yml
```

---

## 6. ACTIVE SCOPE — Phase 1 (build this, nothing else)

**Current active phase: Phase 1.**

Build these modules in order. Finish, review, test, and commit each one before starting the next. Each module below has a purpose, a build spec, a definition of done, and a review checklist the human uses before committing.

### Module 1 — Project scaffold
- **Purpose:** a clean, runnable skeleton.
- **Build:** folder structure from Section 5, dependency setup with uv, `config/settings.py` using pydantic-settings, `.env.example`, a placeholder README, and a shared metadata schema in `src/common/` (fields: company, doc_type, period, section_or_speaker, source_path).
- **Definition of done:** the project installs with one command and imports without error.
- **Review checklist:**
  - [ ] Folder structure matches Section 5.
  - [ ] `uv` install works from a clean state.
  - [ ] Metadata schema exists and is importable.
  - [ ] `.env.example` lists every config value; no secrets committed.

### Module 2 — Ingestion
- **Purpose:** turn PDFs into clean, metadata-rich chunks.
- **Build:** a PyMuPDF loader for the PDFs in `data/`, metadata tagging derived from the filename convention, and a section-aware chunker. Ingestion must be re-runnable without duplicating chunks.
- **Definition of done:** running ingestion over `data/` produces chunks that each carry full metadata.
- **Review checklist:**
  - [ ] Only PyMuPDF is used; no HTML loaders.
  - [ ] Every chunk has company, doc_type, period, and section_or_speaker where available.
  - [ ] Re-running does not create duplicates (verify chunk count is stable).
  - [ ] Chunks are coherent (not cut mid-sentence at arbitrary points).
  - [ ] A quick print of 3 sample chunks looks sensible.

### Module 3 — Indexing
- **Purpose:** build both search indexes from the chunks.
- **Build:** bge embeddings, a persistent Chroma vector store, and a BM25 keyword index, all built from the Module 2 chunks in one pass.
- **Definition of done:** both indexes build and persist; a raw similarity query returns plausible chunks.
- **Review checklist:**
  - [ ] Chroma persists to disk and reloads without rebuilding.
  - [ ] BM25 index covers the same chunks.
  - [ ] Metadata survives into the vector store (filterable).
  - [ ] A test query returns on-topic chunks.

### Module 4 — Retrieval
- **Purpose:** retrieve the right passages, precisely.
- **Build:** a LangChain EnsembleRetriever combining dense and BM25, an optional metadata filter (by company or quarter), and a bge-reranker second stage.
- **Definition of done:** a question returns a ranked, reranked short list of relevant chunks with metadata.
- **Review checklist:**
  - [ ] Hybrid retrieval runs both indexes and merges.
  - [ ] Metadata filter correctly scopes to a company or quarter.
  - [ ] Reranker reorders results (compare before/after on one query).
  - [ ] A KO question does not return PEP/MDLZ chunks when scoped.

### Module 5 — Generation
- **Purpose:** grounded, cited answers, with honest refusal.
- **Build:** a grounding prompt (answer only from context), citation enforcement (every answer points to source chunks), a refusal gate with a tunable confidence threshold, and the Ollama LLM behind a swappable interface.
- **Definition of done:** answerable questions get cited answers; unanswerable ones get refused.
- **Review checklist:**
  - [ ] Every answer includes at least one resolvable citation.
  - [ ] The refusal gate triggers on out-of-corpus questions (test with the refusal questions in the eval set).
  - [ ] The LLM sits behind an interface that could swap to a hosted model.
  - [ ] The answer uses only retrieved context (no outside facts).

### Module 6 — Serving
- **Purpose:** make it an operable service.
- **Build:** a FastAPI endpoint that takes a question and returns answer + citations + confidence, then a Dockerfile (and docker-compose if helpful). Streamlit UI optional.
- **Definition of done:** the API answers a question end to end; the container builds and runs.
- **Review checklist:**
  - [ ] Endpoint returns answer, citations, and a confidence signal.
  - [ ] Swagger docs load.
  - [ ] Docker image builds and the service responds inside the container.

### Module 7 — Evaluation and observability
- **Purpose:** prove it works, with numbers.
- **Build:** a RAGAS harness that runs `eval/question_set.xlsx` and reports faithfulness, answer relevance, and context precision, plus refusal accuracy on the refusal questions. Add structured logging of latency, cost, and top retrieved sources per query.
- **Definition of done:** running the harness prints scores; each query is logged.
- **Review checklist:**
  - [ ] The harness reads the evaluation question set.
  - [ ] It reports faithfulness and refusal accuracy as numbers.
  - [ ] Every query logs latency and retrieved sources.

---

## 7. Phase exit checklist (must be green before advancing)

**Phase 1 is complete only when all of the following hold. Do not begin Phase 2 until the human confirms.**

- [ ] Modules 1 through 7 each pass their review checklist and are committed.
- [ ] The evaluation harness runs end to end on the full question set.
- [ ] Faithfulness clears the agreed threshold (set it explicitly, e.g. 0.85).
- [ ] Refusal accuracy clears the agreed threshold (e.g. all refusal questions refused).
- [ ] The service builds and runs in Docker.
- [ ] README explains how to install, ingest, run, and evaluate.

When these are green, stop and ask before starting Phase 2 (DuckDB + dbt).

---

## 8. Working rhythm (per module)

1. Agent builds one module.
2. Agent reports what it built, how to run it, how to verify it, then stops.
3. Human runs it and reviews the diff against the module's review checklist.
4. Human commits the working module with a clear message (e.g. `feat: ingestion module`).
5. Human pushes to GitHub.
6. Move to the next module.

Commit after every working module so there is always a clean point to return to.

---

## 9. Data conventions

- Source PDFs live in `data/<TICKER>/`.
- Filename convention encodes the metadata: `KO_10-K_FY2025.pdf`, `KO_10-Q_2026Q1.pdf`, `KO_transcript_2026Q1.pdf`, `KO_memo_01.pdf`.
- The ingestion metadata parser reads company, doc_type, and period directly from the filename, so keep names consistent.
- Corpus scope: KO, PEP, MDLZ, two quarters each (Q1 2026 and Q3 2025), roughly 20 documents. Keep it small on purpose so evaluation stays honest.
