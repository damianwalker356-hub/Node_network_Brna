"""Create a portable ZIP from a PyInstaller output directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def create_archive(build_directory: Path, output: Path, platform: str) -> Path:
    if not build_directory.is_dir():
        raise FileNotFoundError(f"desktop build directory does not exist: {build_directory}")
    files = sorted(path for path in build_directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"desktop build directory is empty: {build_directory}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(f"BRNA-Studio-{platform}") / path.relative_to(build_directory))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, default=Path("dist/BRNA-Studio"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", required=True, choices=("windows-x64", "macos", "linux-x64"))
    args = parser.parse_args()
    result = create_archive(args.build_directory, args.output, args.platform)
    print(f"Created {result}")


if __name__ == "__main__":
    main()
