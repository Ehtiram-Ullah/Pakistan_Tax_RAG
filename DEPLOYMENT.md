# Deploy the backend to Vercel

This backend keeps **PyTorch + SentenceTransformers + BAAI/bge-small-en-v1.5**.
Qdrant Cloud holds the vectors and passage text. Gemini generates the answer.
The frontend remains a separate Vercel project.

## 1. Transfer the existing collection (once)

Creating a Qdrant cluster does not populate it. Put the cluster HTTPS endpoint and
API key in the backend's local `.env`, along with the existing Gemini key:

```dotenv
QDRANT_URL=https://YOUR-CLUSTER-ENDPOINT:6333
QDRANT_API_KEY=YOUR-KEY
GEMINI_API_KEY=YOUR-KEY
```

Keep these values private. Do not overwrite your existing `.env` with the example.
Use a key with collection write permissions for migration. A separate read-only
Qdrant key is sufficient for the deployed API.

Stop any API/indexing process using the local database. From this backend folder:

```powershell
.\.venv\Scripts\python.exe -m scripts.migrate_qdrant
.\.venv\Scripts\python.exe -m scripts.migrate_qdrant --apply
```

The first command only inspects. The second copies the collection, verifies every
ID, vector and payload, and checks the total count. It preserves your local data
and refuses to overwrite an existing cloud collection. If interrupted after
collection creation, inspect the partial destination before retrying; there is
deliberately no automatic delete or overwrite mode.

No PDF uploads or re-embedding are required. `storage.sqlite` is not a cloud
snapshot: this script reads the full local Qdrant directory and sends points
through the Qdrant API. Keep the original PDFs and local index as backups.

## 2. Verify the model and cloud retrieval locally

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_model
.\.venv\Scripts\python.exe -m scripts.check_deployment
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The build script downloads a pinned snapshot of the same embedding model, keeps
only its PyTorch weights/tokenizer/configuration, and checks a real 384-dimensional
embedding. Generated model files are ignored by Git. The cloud check does not use
Gemini credits or change the database.

## 3. Import the backend Git repository into Vercel

- Plan: Hobby, for this personal portfolio project.
- Root directory: the backend repository root (the folder containing this file).
- Framework preset: FastAPI.
- Python: 3.12, selected by `.python-version`.
- Vercel resolves dependencies from the `[project]` table in `pyproject.toml`.
  Its `tool.uv.sources` setting selects CPU-only PyTorch on Linux x86-64.
  Commit `uv.lock` alongside the manifest so deployments use the tested resolution.
  `requirements.txt` remains available for local pip-based setup.
- Keep the install and build command overrides empty; the repository provides
  the build hook in `pyproject.toml`.
- Enable Fluid compute / Active CPU. Set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.

Set these **backend project environment variables** before deploying:

| Variable | Value |
| --- | --- |
| `GEMINI_API_KEY` | Your existing Gemini key |
| `QDRANT_URL` | Your Qdrant cluster HTTPS endpoint |
| `QDRANT_API_KEY` | A Qdrant key with read access to the collection |
| `CORS_ORIGINS` | Exact frontend origin, e.g. `https://your-app.vercel.app` (no trailing slash) |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | `1` |
| `GEMINI_MODELS` | Optional comma-separated preferred models; otherwise existing defaults apply |

`vercel.json` selects `iad1` near the N. Virginia database and a 300-second
function duration. Dependencies use a CPU-only Linux PyTorch wheel. PDF parsing
and indexing dependencies are isolated in `requirements-indexing.txt` for local
rebuilds. PDFs, SQLite data, credentials and development files are excluded from
the function bundle. Model weights are downloaded during the build and bundled;
requests never download the model on Vercel.

## 4. Connect and test the frontend

In the frontend Vercel project, set `NEXT_PUBLIC_API_URL` to the deployed backend
origin and redeploy the frontend. Set backend `CORS_ORIGINS` to the actual frontend
production origin and redeploy the backend if it changed. Multiple allowed
origins are comma-separated. A preview frontend needs its own exact allowed origin.
Make sure Vercel Deployment Protection is not blocking the public backend's
research endpoints or browser CORS preflight requests.

Open the backend `/health`, then ask a real question from the frontend. Confirm
retrieval, model-attempt events and cited results. `/health` is liveness only:
it intentionally does not load PyTorch, query Qdrant or call Gemini. Test a cold
request as well as a second request, and inspect Vercel's memory/duration logs.

## What still needs the actual deployment

Local tests cannot prove a Linux Vercel bundle fits or that cold-start memory stays
under Hobby's 2 GB limit. The current platform supports Python large-function
bundles up to 5 GB in beta, with a 300-second Hobby execution limit. The first
deployment must validate installation, bundle size, memory and streaming end to
end. No paid upgrade is assumed by this configuration.

Sources checked October 14, 2026:
- [FastAPI entrypoint and build hook](https://vercel.com/docs/frameworks/backend/fastapi)
- [Python bundling](https://vercel.com/docs/functions/runtimes/python)
- [Current function limits and large-function settings](https://vercel.com/docs/functions/limitations)

Model catalog, cooldown and successful-model caches live within each warm function
instance. Separate instances may choose different first models. CORS controls
browser access; it does not authenticate this public API or prevent direct usage.
