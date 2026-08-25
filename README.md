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

**Phase 1, Modules 1 through 7**: complete.

**Refusal gate hardening — scope-coverage recovery step.** The scope-coverage
check (part of Module 5's refusal gate) no longer refuses the instant it
finds a company or period the retrieved evidence doesn't cover. It now
attempts one bounded recovery first: for each missing (company, period)
pair, it runs one targeted retrieval scoped to that pair, merges the result
into the original candidates, and reranks the merge with the existing
bge-reranker-base cross encoder -- then checks coverage exactly once more
before deciding. If the gap closes, generation proceeds with the merged
evidence; if not, it refuses, naming what's still missing and noting that a
targeted retrieval was already attempted.

Measured on the 15-question evaluation set (`llama3.2:1b`, threshold 0.1,
same judge model, only this change): overall decision accuracy held at
**9/15**, unchanged. One question (Q10) was recovered from a refusal into a
grounded, cited answer. One question (Q06) still refuses, but now for an
accurate reason after a genuine repair attempt, instead of the earlier
run's fabricated citation (the model had attributed a claim to KO that was
actually sourced from an MDLZ passage). One question (Q03) flipped
independently of this change -- confirmed by re-running retrieval for it in
isolation, which returned a byte-identical top score both times; the only
difference was the LLM's own output at `temperature=0.0`, the same
run-to-run nondeterminism already noted elsewhere in this project.
Context precision on the answered questions moved from 0.717 to 0.851, and
answer relevance also rose -- worth noting, but treat both cautiously: only
6 of 15 questions were answered in either run, and a different 6 (Q03 out,
Q10 in), so it's a small and not directly comparable sample.

**CI evaluation gate.** Every push and pull request runs a fast,
deterministic gate (GitHub Actions) that checks pre-generation refusal
accuracy and decision accuracy on the 15-question set without an LLM judge
-- see [CI evaluation gate](#ci-evaluation-gate) below for the two-mode
design, current thresholds, and how to verify it.

## Install

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com)
(for generation, Module 5 onward).

```bash
uv sync
brew install ollama          # or see ollama.com for other platforms
brew services start ollama
ollama pull llama3.2:1b      # or any model; set OLLAMA_MODEL to match
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

### Generation

Generation runs Module 4's retrieval, then a five-stage refusal gate before
handing anything to the LLM. The first two are deterministic pre-generation
checks added later (no LLM call, see the scope-coverage recovery step
above); the rest are Module 5's original three:

1. Evidence-type check: refuses questions asking for live/current market
   data (e.g. a stock price "today"), which this corpus can never contain,
   regardless of retrieval score.
2. Scope-coverage check, with one bounded repair attempt: if the question
   names a company/period the retrieved evidence doesn't cover, one
   targeted retrieval is tried before deciding -- see above.
3. If retrieval's top reranked score doesn't clear `refusal_confidence_threshold`
   (default 0.1, empirically calibrated -- see Module 5 review notes), it
   refuses before calling the LLM at all.
4. The LLM is prompted to answer only from the retrieved passages and to
   respond with a fixed marker if they don't answer the question; if it
   does, that's a refusal too.
5. If the model's answer doesn't cite any real retrieved passage, it can't
   be verified as grounded, so it's treated as a refusal rather than shown.

```bash
uv run python -m src.generation.pipeline "How did operating margin change for Mondelez?"
uv run python -m src.generation.pipeline "What is the capital of France?"
```

The LLM itself sits behind a small interface (`src/generation/llm.py`) so a
hosted model could replace Ollama by adding one new class, with no changes
anywhere else in generation.

### Serving

A FastAPI service wraps retrieval through generation behind one endpoint.

**Locally:**

```bash
uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How did operating margin change for Mondelez?"}'
```

Optional `company` / `period` fields on the request scope retrieval, same as
the CLI's `--company`/`--period` flags. Swagger docs: `http://localhost:8000/docs`.

**In Docker** (app + a sidecar Ollama container, so nothing needs to be
installed on the host beyond Docker itself):

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3.2:1b  # first run only
```

`data/` (chunks, the persisted Chroma/BM25 indexes) is mounted from the host,
not baked into the image, so ingestion and indexing must already have been
run on the host first. Set `APP_PORT` if 8000 is taken on your machine.

### Evaluation

Runs every question in `eval/question_set.xlsx` through the full pipeline,
scores answered questions with RAGAS (faithfulness, answer relevance,
context precision -- judged locally by the same Ollama model and bge
embeddings used elsewhere, no paid API calls), and reports refusal accuracy
on the sheet's dedicated refusal-test questions:

```bash
uv run python -m src.evaluation.harness
```

Every query is logged to `logs/queries.jsonl` (latency, refusal status, top
retrieved sources), refused or not.

A local 1-2B Ollama model is a slow RAGAS judge -- each metric involves
several sequential LLM calls, so a 15-question run can take a couple of
hours; some judge calls will time out or fail to parse regardless (this is
a known limitation of using a small local model as the judge, not a bug --
see Module 7 review notes). `RunConfig(timeout=600, max_workers=2)` in
`src/evaluation/harness.py` is tuned for this; a larger/hosted judge model
would run much faster with the defaults.

### CI evaluation gate

The full RAGAS harness above is too slow (1.5-2.5 hours) to run on every
push, so evaluation is split into two modes:

| | Fast gate (`src/evaluation/fast_gate.py`) | Full harness (`src/evaluation/harness.py`) |
|---|---|---|
| Runs | Every push and PR (`.github/workflows/fast-evaluation-gate.yml`) | Manually only, via `workflow_dispatch` (`.github/workflows/full-ragas-evaluation.yml`) |
| LLM calls | None | Ollama, as both generator and RAGAS judge |
| What it checks | Real retrieval (dense + BM25 + bge-reranker-base) plus the three deterministic, LLM-free pre-generation refusal gates -- evidence-type check, scope-coverage check (with its repair attempt), score-threshold check | Full pipeline through generation, scored with RAGAS (faithfulness, answer relevance, context precision) |
| Speed | Minutes | 1.5-2.5 hours |
| Determinism | Yes -- verified by running it twice locally and diffing the JSON output byte-for-byte | No -- the LLM judge and generation model are not perfectly deterministic even at `temperature=0.0` (documented elsewhere in this README) |

The fast gate stops right before the LLM call, so it measures whether a
question would *reach* generation, not whether the eventual answer is
good -- a question that clears all three gates is reported as "accepted",
not "answered correctly". It calls the exact same gate functions
`src/generation/pipeline.py` calls, in the same order, so it can't silently
drift from what the real pipeline does up to the point it stops. Full
answer quality is still only measured by the slow harness.

**Why retrieval runs for real in CI, not from a fixture:** `data/chroma/`
and `data/bm25/` are gitignored, regenerable build artifacts -- never
committed. `data/processed/chunks.jsonl` (ingestion's output) *is*
committed, so the CI workflow rebuilds the real indexes from it
(`uv run python -m src.indexing.pipeline`) every run, the same command a
developer runs locally. bge-base-en-v1.5 and bge-reranker-base are small,
public, CPU-only, no-auth-required models that genuinely run on a hosted
GitHub Actions runner in a few minutes, so there was no need to fake
retrieval or replay a canned sample of past results -- only the Ollama LLM
judge is the genuinely slow, CI-inappropriate part, and the fast gate never
touches it.

**Thresholds** (`eval/gate_thresholds.json`), set to match the currently
measured state so the gate passes now and fails on a real regression:

```json
{
  "min_refusal_accuracy": 0.66,
  "min_decision_accuracy": 0.66
}
```

Measured locally at the time these were set: `refusal_accuracy` = 2/3
(0.667, one of the three out-of-corpus questions is accepted at the
pre-generation stage -- it's later caught by the LLM's own self-refusal, a
gate this fast mode doesn't exercise) and `decision_accuracy` = 10/15
pre-generation (0.667). 0.66 sits just below both measured values: the
current state passes, but the next-lower discrete outcome for either
metric (1/3 refusal accuracy, or 9/15 decision accuracy) would fail the
build.

**These are not the same measurement as the full-pipeline 9/15 decision
accuracy reported above, and they aren't supposed to match.** The fast
gate stops before generation, so its `decision_accuracy` and
`refusal_accuracy` only count what the three deterministic pre-generation
gates resolve. Concretely, the fast gate's refusal accuracy is 2/3, not
3/3, because Q15 ("What is PepsiCo's detailed plan for 2030?") clears all
three deterministic gates and is reported as "accepted" here -- in the
full pipeline it's still correctly refused, just later, when the model
itself declines to answer via its own self-refusal marker, a gate this
fast mode deliberately never runs. If these two numbers ever look like
they contradict each other, they don't -- they're measuring different
checkpoints in the same pipeline.

```bash
uv run python -m src.evaluation.fast_gate
```

Prints a summary, writes it to `eval/fast_gate_result.json`, and exits 1 if
either threshold isn't met.
