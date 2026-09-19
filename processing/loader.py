from llama_index.core import Document
from parser import extract_text



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


if __name__ == "__main__":
    docuemnts = load_pdf("data/raw/income_tax_ordinance_2026.pdf")
    print(docuemnts[0])