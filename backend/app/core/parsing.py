"""Extract plain text from supported document formats."""
import io


def extract_text(raw: bytes, ext: str) -> str:
    if ext == ".txt" or ext == ".md":
        return raw.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported extension: {ext}")
