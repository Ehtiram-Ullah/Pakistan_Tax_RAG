from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generation.generator import generate_answer


app = FastAPI(
    title="Pakistan Tax Law RAG",
    description="RAG system for Pakistani tax legislation",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Pakistan Tax Law RAG API is running"
    }


@app.post("/ask")
def ask_question(data: Question):

    result = generate_answer(data.question)

    return {
        "question": data.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }