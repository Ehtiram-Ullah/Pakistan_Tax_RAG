# TaxLens API

Evidence-grounded Pakistani tax legislation research, powered by local semantic retrieval and a configurable Gemini fallback chain.

## Setup

From this directory, using a Python environment compatible with the pinned packages:

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `GEMINI_API_KEY` and an ordered `GEMINI_MODELS` list in `.env`. The example defaults are configuration examples; availability depends on your Google account. Your configured models lead the first request, followed by compatible text-generation models discovered from your account. Discovery is lazy, reads at most one 100-model catalog page, and is cached for 15 minutes. Media, embedding, robotics, and specialized tool models are excluded. Up to 16 distinct candidates are attempted, with a 20-second timeout per attempt, a 150-second generation budget, and SDK retries disabled. Discovery and provider calls never occur at import or health-check time. A catalog entry is not a guarantee of account quota or access.

Start the API from this directory:

```sh
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use **one worker** for the embedded Qdrant store. Do not run indexing and the API against that store simultaneously. For multiple worker processes, migrate to a Qdrant server. The first search can take longer while the BGE model loads or downloads; later searches reuse it. Retrieval is serialized to protect the local connection.

## Corpus

The existing index is under `storage/qdrant`. To intentionally rebuild it from the PDFs in `data/raw`, stop the API first, then run:

```sh
python -m storage.qdrant.vector_store
```

This existing index builder replaces the collection. Confirm the source documents first. Document filenames do not prove currency; verify effective dates, amendments, and the relevant tax year against original publications.

`QDRANT_PATH` optionally sets a different database location for the API. The default path resolves relative to the backend module, not the working directory. The existing index builder uses `storage/qdrant` relative to its working directory.

## API

- `GET /health`: API liveness, whether a key is configured, and configured model names. It does not prove provider access or index readiness.
- `POST /ask`: compatible non-streaming research endpoint.
- `POST /ask/stream`: live NDJSON events for retrieval, attempts, failures, completion, and errors.
- `GET /docs`: interactive API documentation.

Both research endpoints accept `{ "question": "..." }`, trim whitespace, and validate a 3–2,000 character limit. Sources include an ID, filename, page, semantic similarity, and the actual retrieved excerpt. Answers are instructed to cite those IDs and acknowledge missing evidence. No evidence means no generation call. Citations are model-generated and still require verification.

Provider errors are classified into safe user-facing reasons. Raw provider payloads, API credentials, and prompts are not returned to the client. CORS origins are configured through `CORS_ORIGINS`. The API does not include authentication or rate limiting; add them before a public deployment with paid credentials.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests use fake providers and retrieval results: no API credits, downloads, or corpus mutations. Coverage includes fallback ordering and provenance, exhausted and empty responses, empty retrieval, input validation, streaming framing, safe errors, and compatibility with `/ask`.

## Architecture

```text
Next.js UI → FastAPI NDJSON stream
                 ↓
       BGE embeddings → Qdrant passages
                 ↓
       Gemini model 1 → model 2 → …
                 ↓
       Cited answer + evidence + model provenance
```

Embedding/database resources are loaded lazily, reused between requests, and closed on application shutdown. A browser cancellation closes its stream; an in-flight synchronous provider call may finish before the server notices cancellation. Its timeout limits that work. The browser allows five minutes including retrieval, discovery, and generation.

Default routing preserves the project's original Flash model order: `gemini-3.8-flash`, `gemini-3.7-flash`, then `gemini-3.5-flash`. Model identifiers were checked against [Google's model catalog](https://ai.google.dev/gemini-api/docs/models) on October 6, 2026; account access and quota still require a real-key check.


## Efficient fallback routing

The last successful model is tried first on the next request. Models that return 429 enter a 60-second cooldown, 404 a 15-minute cooldown, and transient server failures a 30-second cooldown. Skipped models are reported as `model_skipped` events, so the UI explains the decision. Cooldowns and catalog caches are isolated by a fingerprint of the API key and cleared by a server restart. A failed catalog refresh temporarily reuses stale entries or a built-in list of additional text models; it does not block your configured choices.

New streaming events: `discovery_started`, `discovery_complete`, and `model_skipped`. All generation failures still report a reason and the next candidate. The fallback list is finite and time-bounded: if every compatible model lacks quota or access, the app returns a useful error rather than looping forever. Update project quota/billing or retry after the provider limit resets in that case.
