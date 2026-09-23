from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


from processing.loader import load_pdf, chunk_documents

# from 

from processing.embedding import embed_documents





COLLECTION_NAME = "pakistan_tax_law"

PDF_FILES = [
    "data/raw/income_tax_ordinance_2026.pdf",
    "data/raw/sales_tax_act_2026.pdf",
    "data/raw/finance_act_2026.pdf",
]


def build_vector_store():
    client = QdrantClient(path="storage/qdrant")

    # Remove old collection so we don't mix old/new data
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    point_id = 0

    for pdf_path in PDF_FILES:
        print(f"\nProcessing: {pdf_path}")

        documents = load_pdf(pdf_path)
        nodes = chunk_documents(documents)
        embeddings = embed_documents(nodes)

        points = []

        for node, embedding in zip(nodes, embeddings):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": node.text,
                        "source": node.metadata["source"],
                        "page": node.metadata["page"],
                    }
                )
            )

            point_id += 1

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"Inserted {len(points)} chunks.")

    print(f"\nTotal chunks: {point_id}")

    client.close()


if __name__ == "__main__":
    build_vector_store()