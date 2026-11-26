import zipfile

import pytest

from scripts.archive_desktop import create_archive


@pytest.mark.req("R-STUDIO-03")
def test_desktop_archive_contains_only_packaged_application_files(tmp_path):
    build = tmp_path / "dist" / "BRNA-Studio"
    (build / "web").mkdir(parents=True)
    (build / "BRNA-Studio").write_bytes(b"launcher")
    (build / "web" / "index.html").write_text("studio", encoding="utf-8")
    output = tmp_path / "dist" / "brna-studio-linux-x64.zip"

    created = create_archive(build, output, "linux-x64")

    assert created == output
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "BRNA-Studio-linux-x64/BRNA-Studio",
            "BRNA-Studio-linux-x64/web/index.html",
        ]


@pytest.mark.req("R-STUDIO-03")
def test_desktop_archive_rejects_missing_or_empty_builds(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_archive(tmp_path / "missing", tmp_path / "app.zip", "linux-x64")

    empty_build = tmp_path / "empty"
    empty_build.mkdir()
    with pytest.raises(ValueError, match="empty"):
        create_archive(empty_build, tmp_path / "app.zip", "linux-x64")
