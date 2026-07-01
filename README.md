# DocuTrust

Self-correcting Retrieval-Augmented Generation platform that ingests corporate policy documents, validates every retrieval with a cross-encoder, falls back to a web search when corpus coverage is thin, and refuses to answer without strict source citations.

## Why

Generic RAG chatbots hallucinate, especially on dense policy packets where the LLM confidently invents language that isn't in the source. DocuTrust adds a **Corrective RAG** (CRAG) loop:

1. **Retrieve** — semantic search over structural chunks.
2. **Grade** — a local cross-encoder scores every chunk. Chunks below the relevance threshold are dropped.
3. **Decide** — if any chunks survive, the LLM answers with citations. If none survive, the query is **rewritten** and a **web search** runs.
4. **Generate** — the LLM is forced to cite `[doc_id:chunk_id]` markers in its output. Anything uncited is rejected and returned as a refusal.

Every run writes a structured trace to MongoDB `trace_logs` so you can audit what the system retrieved, what it graded out, and what it cited.

## Stack

- **FastAPI** + **Uvicorn** for the API.
- **LangGraph** for the CRAG state machine.
- **MongoDB 8.x** (local `mongod` for dev, Atlas for prod).
- **Pluggable vector store** — `LocalFaissStore` (dev) or `AtlasVectorStore` (`$vectorSearch`, prod).
- **OpenAI** for chat + embeddings.
- **sentence-transformers** for the offline cross-encoder grader.
- **Streamlit** for the demo UI.

## Setup

```bash
brew install python@3.11              # one-time, only if you don't have 3.11+
cd docutrust
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                  # then set OPENAI_API_KEY
./scripts/start_mongo.sh              # local mongod
python -m scripts.seed_corpus         # ingests data/sample_policy.pdf
uvicorn app.main:app --reload --port 8000
```

In another shell:

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py
```

## Endpoints

| Method | Path              | Purpose                                            |
|--------|-------------------|----------------------------------------------------|
| GET    | `/healthz`        | Liveness check                                     |
| POST   | `/clients`        | Create a client profile                            |
| GET    | `/clients`        | List clients                                       |
| POST   | `/ingest`         | Upload a PDF / DOCX / MD for a client              |
| POST   | `/ask`            | Ask a question; runs the CRAG graph                |
| GET    | `/ask/{trace_id}` | Retrieve a previous trace                          |

## Environment variables

See `.env.example`. The two most important:

- `VECTOR_BACKEND` — `faiss` (default) or `atlas`.
- `OPENAI_API_KEY` — required.

## Three demo flows

1. **Happy path** — ask a question answered by the seed corpus. Expect an answer with ≥1 citation, no `web_search` step in the trace.
2. **Low relevance → rewrite → web** — ask something outside the corpus. Expect a `rewrite` step and a `web_search` step in the trace; citations point to web sources.
3. **Refusal** — ask a question the LLM is tempted to answer from general knowledge with no sources. Expect a "rejected, no citations" event and a refused answer.

## Tests

```bash
pytest -q
```

The graph tests stub the LLM and the grader, so they run with no network and no model downloads.

## Project layout

See the plan file at `/Users/anurag/.claude/plans/jiggly-questing-valiant.md` for the full layout and design notes.
