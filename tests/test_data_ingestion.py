import base64
import zipfile
from io import BytesIO

import pytest
from reportlab.pdfgen import canvas

from brna.data_ingestion import extract_training_text


@pytest.mark.req("R-UPLOAD-01")
def test_extract_training_text_supports_text_and_docx():
    text, kind = extract_training_text(
        "lesson.md",
        base64.b64encode(b"# A title\n\nA training corpus.").decode("ascii"),
    )
    assert kind == "text"
    assert text == "# A title A training corpus."

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p></w:body></w:document>',
        )
    text, kind = extract_training_text(
        "lesson.docx",
        base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    assert kind == "docx"
    assert "First paragraph." in text
    assert "Second paragraph." in text


@pytest.mark.req("R-UPLOAD-01")
def test_extract_training_text_rejects_invalid_and_unsupported_inputs():
    with pytest.raises(ValueError, match="base64"):
        extract_training_text("lesson.txt", "not base64!")
    with pytest.raises(ValueError, match="not supported"):
        extract_training_text(
            "legacy.doc",
            base64.b64encode(b"legacy").decode("ascii"),
        )


@pytest.mark.req("R-UPLOAD-01")
def test_extract_training_text_supports_pdf():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "PDF training corpus")
    pdf.save()
    text, kind = extract_training_text(
        "lesson.pdf",
        base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    assert kind == "pdf"
    assert "PDF training corpus" in text
