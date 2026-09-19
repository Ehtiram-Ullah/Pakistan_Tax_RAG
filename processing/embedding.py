"""
It provides pre-trained Machine Learning models specifically fine-tuned to map whole sentences or paragraphs into dense vector spaces
(embeddings) so that semantically similar texts end up close to each other.
"""
from sentence_transformers import SentenceTransformer

from processing.loader import load_pdf, chunk_documents

"""
open-source model hosted on Hugging Face created by the Beijing Academy of Artificial Intelligence (BAAI).
bge: Stands for BAAI General Embedding.

small: Indicates a lightweight version (~33 million parameters), making it fast to run even on a standard CPU.

en: Fine-tuned specifically for English text.

v1.5: Version tag featuring improved retrieval performance and distance matching.
"""
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_documents(nodes):

    texts = [node.text for node in nodes]

    """
    normalize_embeddings=True: Scales every output vector so its length (L2 norm) equals 1.
    Why this matters: When embeddings are normalized, computing Cosine Similarity (how similar two texts are) simplifies to a fast,
    simple Dot Product calculation.
    """
    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


#Testing
if __name__ == "__main__":
    documents = load_pdf("data/raw/income_tax_ordinance_2026.pdf")

    nodes =chunk_documents(documents)

    embeddings = embed_documents(nodes)

    print("Number of chunks: ", len(nodes))
    print("Embedding shape: ",embeddings.shape)
    print(embeddings[0])