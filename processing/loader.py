from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from processing.parser import extract_text



def load_pdf(pdf_path):
    pages = extract_text(pdf_path)

    documents = []

    for page in pages:
        document = Document(
            text = page["text"],
            metadata = {
                "source": pdf_path,
                "page": page["page"]
            }
        )

        documents.append(document)
    
    return documents

def chunk_documents(documents):
    splitter = SentenceSplitter(
        # 512 tokens per chunk
        chunk_size=512,
        # means the next chunk repeats roughly 50 tokens from the previous one
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes



if __name__ == "__main__":
    documents = load_pdf(
            "data/raw/income_tax_ordinance_2026.pdf"
        )

    nodes = chunk_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(nodes)}")

    print("\n--- LAST CHUNK ---")
    print(nodes[-1])