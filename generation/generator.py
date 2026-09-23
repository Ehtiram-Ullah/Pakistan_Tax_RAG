import os
import time

from dotenv import load_dotenv
from google import genai

from retrieval import search


load_dotenv()

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]
for model in gemini_client.models.list():
    MODELS.append(model.name.replace("models/",""))



def generate_answer(question, limit=5):

    results = search(question, limit)

    context = ""
    sources = []

    for result in results:

        sources.append({
            "document": result.payload["source"],
            "page": result.payload["page"],
            "score": result.score
        })

        context += f"""
SOURCE: {result.payload["source"]}
PAGE: {result.payload["page"]}

{result.payload["text"]}

---
"""

    prompt = f"""
You are a Pakistani tax-law research assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent tax rates, laws, sections, or conditions.
3. If the context contains conflicting provisions, explain the
   difference instead of choosing one without evidence.
4. If the context is insufficient, say so.
5. Keep the answer concise.
6. Mention the relevant document and page numbers.

CONTEXT:
{context}

QUESTION:
{question}
"""

    last_error = None

    for model in MODELS:

      

        try:

            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt
            )

            return {
                "answer": response.text,
                "sources": sources
            }

        except Exception as error:

            last_error = error

            print(
                f"{model} failed "
                f"error: {error}"
            )

            time.sleep(2)

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    )


if __name__ == "__main__":

    question = "What is the tax on a motor vehicle between 1601cc and 1800cc?"

    result = generate_answer(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print(
            source["document"],
            "Page:", source["page"],
            "Score:", source["score"]
        )