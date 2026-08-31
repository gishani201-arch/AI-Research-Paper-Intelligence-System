import fitz


def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from all pages of an uploaded PDF.
    """

    pdf_bytes = uploaded_file.read()

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages_text = []

    for page in pdf_document:
        text = page.get_text()
        pages_text.append(text)

    pdf_document.close()

    return "\n".join(pages_text)