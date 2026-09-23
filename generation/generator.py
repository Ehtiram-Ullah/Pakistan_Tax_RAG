import os

from dotenv import load_dotenv
from google import genai

from retrieval import search

load_dotenv()


client = genai.Client(
    api_key = os.getenv("GEMINI_API_KEY")
)


def generate_answer(question,limit =5):
    results = search(question,limit)

    context = ""

    for result in results:
        context += f"""
        SOURCE: {result.payload["source"]},
        PAGE: {result.payload["page"]}

        {result.payload["text"]}

        --------
        """
    prompt = f"""
        You are a Pakistani tax-law assistant.

        Answer the user's question using ONLY the provided context.

        Rules:
        - Do not use outside knowledge.
        - If the context does not contain enough information, say:
        "I could not find enough information in the retrieved documents."
        - Do not invent tax rates, conditions, or legal provisions.
        - Give a concise explanation.
        - Mention the relevant page numbers.

        CONTEXT:
        {context}

        USER QUESTION:
        {question}
        """
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    question = "What is the tax on a motor vehicle between 1601cc and 1800cc?"

    answer = generate_answer(question)

    print("\nANSWER:")
    print(answer)

    client.close()