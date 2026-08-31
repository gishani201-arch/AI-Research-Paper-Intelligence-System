import re


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """
    Cleans text extracted from a research paper PDF.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces before new lines
    text = re.sub(r" +\n", "\n", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated spaces after punctuation
    text = re.sub(r"([.!?,;:]) {2,}", r"\1 ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ---------------------------------------------------------
# CREATE TEXT CHUNKS
# ---------------------------------------------------------

def create_chunks(text, chunk_size=1200, overlap=200):
    """
    Splits text into overlapping chunks.

    chunk_size:
        Approximate number of characters in each chunk.

    overlap:
        Number of characters shared between consecutive chunks.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        # Extract chunk
        chunk = text[start:end].strip()

        if chunk:

            # Try to end the chunk at a sentence boundary
            if end < text_length:

                last_period = max(
                    chunk.rfind(". "),
                    chunk.rfind("? "),
                    chunk.rfind("! ")
                )

                # Only adjust if a reasonable sentence boundary exists
                if last_period > chunk_size * 0.6:

                    chunk = chunk[:last_period + 1].strip()

                    end = start + len(chunk)

            chunks.append(chunk)

        # Move forward while keeping overlap
        next_start = end - overlap

        # Prevent infinite loops
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks