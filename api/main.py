import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from generation.generator import ResearchError, configured_models, generate_answer, stream_answer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    yield
    from retrieval import close_resources
    close_resources()


app = FastAPI(title="TaxLens · Pakistan Tax Research", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "generation_configured": bool(os.getenv("GEMINI_API_KEY")), "models": configured_models()}


@app.post("/ask")
def ask_question(data: Question):
    try:
        return generate_answer(data.question)
    except ResearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.error("Research failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Research is temporarily unavailable. Check the server configuration and document index.") from error


@app.post("/ask/stream")
def ask_stream(data: Question):
    def events():
        try:
            for event in stream_answer(data.question):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except ResearchError as error:
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except Exception as error:
            logger.error("Streaming research failed: %s", type(error).__name__)
            yield json.dumps({"type": "error", "message": "Research is temporarily unavailable. Check the server configuration and document index."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
