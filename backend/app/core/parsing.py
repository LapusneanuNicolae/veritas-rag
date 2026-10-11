"""Extract plain text from supported document formats."""
import io


def extract_text(raw: bytes, ext: str) -> str:
    if ext == ".txt" or ext == ".md":
        # utf-8-sig strips a leading byte-order mark (BOM) if present
        return raw.decode("utf-8-sig", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported extension: {ext}")
