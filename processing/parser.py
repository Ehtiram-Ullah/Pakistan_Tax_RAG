
# pymupdf provides tools to open, edit, and extract data from PDF files.
import pymupdf

def extract_text(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        pages.append(
            {
                "page": page_number+1,
                "text": page.get_text()
            }
        )


    # - Document Strucutre Informaiton -
    # page = document[20]

    # blocks = page.get_text("blocks")

    # for block in blocks:
    #     print(block[:4])
    #     print(block[4])
    #     print("---")

    # for page_number in [20, 21, 50, 100, 140, 200]:
    #     page = document[page_number]

    #     print(f"\n===== PAGE {page_number + 1} =====")

    #     for block in page.get_text("blocks"):
    #         x0, y0, x1, y1, text, *_ = block
    #         print(f"y={y0:.1f} → {y1:.1f} | {text[:80]!r}")

    document.close()
    return pages



# TESTING
if __name__ == "__main__":
    print(" ---- Incom Tax ------")
    pagesIncomTax = extract_text("data/raw/income_tax_ordinance_2026.pdf")


    print(" ---- Finance Act ------")
    pagesFinanceAct = extract_text("data/raw/finance_act_2026.pdf")


    print(" ---- Sales Tax ------")
    pagesSalesTax = extract_text("data/raw/sales_tax_act_2026.pdf")

    # for page in pages[20:25]:
    #     print(f"\n--- PAGE {page['page']} ---")
    #     print(page["text"][:3000])