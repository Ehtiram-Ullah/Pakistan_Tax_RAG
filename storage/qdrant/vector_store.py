from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


from processing.loader import load_pdf, chunk_documents

# from 

from processing.embedding import embed_documents

client = QdrantClient(path="storage/qdrant")

collection_name = "pakistan_tax_law"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


documents = load_pdf(
    "data/raw/income_tax_ordinance_2026.pdf"
)

nodes = chunk_documents(documents)

embeddings = embed_documents(nodes)

points = []


for i, (node, embedding) in enumerate(zip(nodes, embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": node.text,
                "source": node.metadata["source"],
                "page": node.metadata["page"]
            }
        )
    )
    

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Inserted {len(points)} chunks into Qdrant.")



