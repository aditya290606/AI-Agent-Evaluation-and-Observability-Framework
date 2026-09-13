"""Archive validation and ZipSlip security protection for uploaded agent projects."""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union


class ArchiveSecurityError(Exception):
    """Raised when an uploaded archive violates security constraints."""
    pass


# Backward compatible alias
ArchiveValidationError = ArchiveSecurityError


class ArchiveValidator:
    """Validates and safely unpacks uploaded agent code packages."""

    def __init__(
        self,
        max_uncompressed_bytes: int = 50 * 1024 * 1024,   # 50 MB
        max_files: int = 500,
        max_file_count: Optional[int] = None,
        allowed_extensions: Optional[List[str]] = None,
        blocked_extensions: Optional[List[str]] = None,
    ):
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_file_count = max_file_count or max_files
        self.allowed_extensions = allowed_extensions
        self.blocked_extensions = [e.lower() for e in (blocked_extensions or [
            ".exe", ".dll", ".so", ".dylib", ".bin", ".com", ".scr", ".msi", ".bat", ".cmd", ".ps1", ".vbs"
        ])]

    def validate_and_extract(self, archive_path: Union[str, Path], extract_to: Union[str, Path]) -> Path:
        """High-level method to validate and extract archive, returning destination Path."""
        archive_p = Path(archive_path).resolve()
        target_p = Path(extract_to).resolve()

        if archive_p.is_dir():
            self.validate_directory(str(archive_p))
            if archive_p != target_p:
                shutil.copytree(archive_p, target_p, dirs_exist_ok=True)
            return target_p

        if zipfile.is_zipfile(archive_p):
            self.validate_and_extract_zip(str(archive_p), str(target_p))
            return target_p

        raise ArchiveSecurityError(f"Unsupported archive format for file: '{archive_p.name}'. Please upload a ZIP archive.")

    def validate_and_extract_zip(self, zip_path: str, extract_to: str) -> Tuple[bool, List[str]]:
        """
        Safely inspects and extracts a ZIP file, preventing ZipSlip path traversal vulnerabilities.
        Returns: (success, list_of_extracted_relative_paths)
        """
        extract_to_path = Path(extract_to).resolve()
        extract_to_path.mkdir(parents=True, exist_ok=True)

        if not zipfile.is_zipfile(zip_path):
            raise ArchiveSecurityError(f"File '{zip_path}' is not a valid ZIP archive.")

        total_size = 0
        extracted_files = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            if len(members) > self.max_file_count:
                raise ArchiveSecurityError(
                    f"Archive exceeds maximum file count limit ({len(members)} > {self.max_file_count})."
                )

            for member in members:
                # 1. ZipSlip Path Traversal Detection
                target_path = (extract_to_path / member.filename).resolve()
                try:
                    target_path.relative_to(extract_to_path)
                except ValueError:
                    raise ArchiveSecurityError(
                        f"ZipSlip security violation detected! Member '{member.filename}' attempts path traversal outside destination."
                    )

                # 2. Blocked binary executable extensions
                ext = Path(member.filename).suffix.lower()
                if ext in self.blocked_extensions:
                    raise ArchiveSecurityError(
                        f"Blocked file extension detected: '{member.filename}' ({ext})"
                    )

                # 3. Accumulate uncompressed size to prevent zip-bomb DoS
                total_size += member.file_size
                if total_size > self.max_uncompressed_bytes:
                    raise ArchiveSecurityError(
                        f"Archive uncompressed size exceeds maximum threshold ({total_size / (1024*1024):.1f}MB > {self.max_uncompressed_bytes / (1024*1024):.1f}MB)."
                    )

            # Safe extraction
            for member in members:
                zf.extract(member, path=extract_to_path)
                if not member.is_dir():
                    rel_p = str(Path(member.filename)).replace("\\", "/")
                    extracted_files.append(rel_p)

        return True, extracted_files

    def validate_directory(self, dir_path: str) -> Tuple[bool, List[str]]:
        """Validates an existing project directory structure for file count, size, and prohibited extensions."""
        p = Path(dir_path).resolve()
        if not p.is_dir():
            raise ArchiveSecurityError(f"Path '{dir_path}' is not a valid directory.")

        total_size = 0
        valid_files = []

        for root, dirs, files in os.walk(p):
            # Skip virtual environments and git directories
            dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"]]

            for file in files:
                fpath = Path(root) / file
                ext = fpath.suffix.lower()
                if ext in self.blocked_extensions:
                    raise ArchiveSecurityError(
                        f"Blocked file extension detected in project: '{file}' ({ext})"
                    )

                f_size = fpath.stat().st_size
                total_size += f_size
                valid_files.append(str(fpath.relative_to(p)).replace("\\", "/"))

                if len(valid_files) > self.max_file_count:
                    raise ArchiveSecurityError(f"Project directory exceeds maximum file count limit ({len(valid_files)} > {self.max_file_count}).")

                if total_size > self.max_uncompressed_bytes:
                    raise ArchiveSecurityError(f"Project total size exceeds limit ({total_size / (1024*1024):.1f}MB > {self.max_uncompressed_bytes / (1024*1024):.1f}MB).")

        return True, valid_files
