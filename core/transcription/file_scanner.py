from __future__ import annotations

from pathlib import Path


class FileScanner:

    def scan_directory(
        self, directory: Path, extensions: list[str], recursive: bool = False
    ) -> list[Path]:
        files = []
        patterns = [f"*{ext}" for ext in extensions]
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        return sorted(files)

    def count_files(
        self, directory: Path, extensions: list[str], recursive: bool = False
    ) -> int:
        return len(self.scan_directory(directory, extensions, recursive))
