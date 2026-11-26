"""Local text extraction for uploaded training corpora."""

from __future__ import annotations

import base64
import binascii
import io
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree

MAX_UPLOAD_BYTES = 20_000_000
MAX_EXTRACTED_CHARACTERS = 50_000


def extract_training_text(filename: str, encoded_content: str) -> tuple[str, str]:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename is required")
    if not isinstance(encoded_content, str):
        raise TypeError("file content must be base64 encoded")
    try:
        content = base64.b64decode(encoded_content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("file content must be valid base64") from exc
    if not content:
        raise ValueError("uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded file must not exceed {MAX_UPLOAD_BYTES} bytes")

    extension = PurePosixPath(filename).suffix.casefold()
    if extension in {".txt", ".md", ".markdown"}:
        text = content.decode("utf-8")
        kind = "text"
    elif extension in {".doc", ".rtf"}:
        raise ValueError(f"{extension} uploads are not supported; use .docx, .pdf, .txt, or .md")
    elif extension == ".docx":
        text = _extract_docx(content)
        kind = "docx"
    elif extension == ".pdf":
        text = _extract_pdf(content)
        kind = "pdf"
    else:
        raise ValueError("unsupported file type; use .docx, .pdf, .txt, or .md")

    normalized = " ".join(text.replace("\r", "\n").split())
    if len(normalized) < 2:
        raise ValueError("uploaded file did not contain enough text")
    if len(normalized) > MAX_EXTRACTED_CHARACTERS:
        normalized = normalized[:MAX_EXTRACTED_CHARACTERS]
    return normalized, kind


def _extract_docx(content: bytes) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            document = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("uploaded DOCX file is invalid") from exc
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as exc:
        raise ValueError("uploaded DOCX document XML is invalid") from exc

    paragraphs = []
    for paragraph in root.iter(namespace + "p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter()
            if node.tag in {namespace + "t", namespace + "tab"}
        )
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise RuntimeError("PDF support requires the pinned pypdf dependency") from exc
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"uploaded PDF file is invalid: {exc}") from exc
    return "\n".join(pages)
