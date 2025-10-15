"""
File Manager for SecureAI Personal Assistant
Handles file operations with sandboxing and security controls
"""

import os
import logging
import hashlib
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import mimetypes
import json
from datetime import datetime


@dataclass
class FileInfo:
    """File information structure"""
    path: str
    name: str
    size: int
    mime_type: str
    last_modified: datetime
    is_directory: bool
    hash: Optional[str] = None


class FileManager:
    """Manages file operations with security sandboxing"""

    def __init__(self, config: Dict):
        self.config = config.get("integrations", {}).get("files", {})
        self.enabled = self.config.get("enabled", True)

        # Security settings
        self.allowed_extensions = set(self.config.get("allowed_extensions", [
            ".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".py", ".html"
        ]))
        self.max_file_size = self._parse_size(self.config.get("max_file_size", "10MB"))

        # Sandbox setup
        self.sandbox_root = Path(self.config.get("sandbox_root", "./data/sandbox"))
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        # Blocked paths and patterns
        self.blocked_patterns = [
            "credentials.json", "token.json", ".env", "config.yaml",
            "*.key", "*.pem", "*.p12", "password*", "secret*"
        ]

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"File manager initialized with sandbox: {self.sandbox_root}")

    def _parse_size(self, size_str: str) -> int:
        units = {
            'B': 1,
            'KB': 1024,
            'K': 1024,
            'MB': 1024 * 1024,
            'M': 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
            'G': 1024 * 1024 * 1024
        }

        size_str = size_str.strip().upper()

        num_part = ""
        unit_part = ""
        for char in size_str:
            if char.isdigit() or char == '.':
                num_part += char
            else:
                unit_part += char

        if unit_part not in units:
            raise ValueError(f"Unknown size unit: {unit_part}")

        return int(float(num_part) * units[unit_part])

    def _is_safe_path(self, path: Union[str, Path]) -> bool:
        """Check if path is safe and within sandbox"""
        try:
            path = Path(path).resolve()
            sandbox_root = self.sandbox_root.resolve()

            # Check if path is within sandbox
            try:
                path.relative_to(sandbox_root)
            except ValueError:
                self.logger.warning(f"Path outside sandbox: {path}")
                return False

            # Check blocked patterns
            path_str = str(path).lower()
            for pattern in self.blocked_patterns:
                if pattern.replace("*", "") in path_str:
                    self.logger.warning(f"Blocked pattern detected: {pattern}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Path safety check failed: {e}")
            return False

    def _is_allowed_extension(self, path: Union[str, Path]) -> bool:
        """Check if file extension is allowed"""
        suffix = Path(path).suffix.lower()
        return suffix in self.allowed_extensions or len(self.allowed_extensions) == 0

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return "unknown"

    def _get_file_info(self, path: Path) -> FileInfo:
        """Get detailed file information"""
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "application/octet-stream"

        return FileInfo(
            path=str(path),
            name=path.name,
            size=stat.st_size,
            mime_type=mime_type,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            is_directory=path.is_dir(),
            hash=self._calculate_file_hash(path) if path.is_file() else None
        )

    def list_files(self, directory: Optional[str] = None, 
                   recursive: bool = False, include_hidden: bool = False) -> List[FileInfo]:
        """
        List files in directory

        Args:
            directory: Directory path (relative to sandbox root)
            recursive: Whether to list recursively
            include_hidden: Whether to include hidden files

        Returns:
            List of FileInfo objects
        """
        if not self.enabled:
            return []

        try:
            if directory:
                target_dir = self.sandbox_root / directory
            else:
                target_dir = self.sandbox_root

            if not self._is_safe_path(target_dir):
                return []

            if not target_dir.exists() or not target_dir.is_dir():
                self.logger.warning(f"Directory not found: {target_dir}")
                return []

            files = []

            if recursive:
                pattern = "**/*" if include_hidden else "**/[!.]*"
                paths = target_dir.rglob(pattern)
            else:
                pattern = "*" if include_hidden else "[!.]*"
                paths = target_dir.glob(pattern)

            for path in paths:
                if path.is_file() and self._is_allowed_extension(path):
                    try:
                        file_info = self._get_file_info(path)
                        files.append(file_info)
                    except Exception as e:
                        self.logger.warning(f"Failed to get info for {path}: {e}")
                elif path.is_dir() and recursive:
                    try:
                        file_info = self._get_file_info(path)
                        files.append(file_info)
                    except Exception as e:
                        self.logger.warning(f"Failed to get info for directory {path}: {e}")

            self.logger.info(f"Listed {len(files)} files from {target_dir}")
            return files

        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            return []

    def read_file(self, file_path: str, encoding: str = 'utf-8', 
                  require_confirmation: bool = True) -> Optional[str]:
        """
        Read file contents

        Args:
            file_path: Path to file (relative to sandbox root)
            encoding: Text encoding to use
            require_confirmation: Whether to require user confirmation

        Returns:
            File contents as string or None if failed
        """
        if not self.enabled:
            return None

        try:
            full_path = self.sandbox_root / file_path

            if not self._is_safe_path(full_path):
                return None

            if not full_path.exists() or not full_path.is_file():
                self.logger.warning(f"File not found: {full_path}")
                return None

            if not self._is_allowed_extension(full_path):
                self.logger.warning(f"File extension not allowed: {full_path}")
                return None

            # Check file size
            if full_path.stat().st_size > self.max_file_size:
                self.logger.warning(f"File too large: {full_path} ({full_path.stat().st_size} bytes)")
                return None

            # Security check
            if require_confirmation:
                self.logger.info(f"Reading file: {full_path}")
                # In a real implementation, this would prompt the user

            # Read file
            with open(full_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()

            self.logger.info(f"Read {len(content)} characters from {full_path}")
            return content

        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8',
                   require_confirmation: bool = True) -> bool:
        """
        Write content to file

        Args:
            file_path: Path to file (relative to sandbox root)
            content: Content to write
            encoding: Text encoding to use
            require_confirmation: Whether to require user confirmation

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            full_path = self.sandbox_root / file_path

            if not self._is_safe_path(full_path):
                return False

            if not self._is_allowed_extension(full_path):
                self.logger.warning(f"File extension not allowed: {full_path}")
                return False

            # Check content size
            content_size = len(content.encode(encoding))
            if content_size > self.max_file_size:
                self.logger.warning(f"Content too large: {content_size} bytes")
                return False

            # Security check - always require confirmation for writes
            if require_confirmation:
                self.logger.warning(f"Writing to file: {full_path}")
                # In a real implementation, this would prompt the user
                # For now, we'll assume confirmation is given

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(full_path, 'w', encoding=encoding) as f:
                f.write(content)

            self.logger.info(f"Wrote {len(content)} characters to {full_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            return False

    def delete_file(self, file_path: str, require_confirmation: bool = True) -> bool:
        """
        Delete a file

        Args:
            file_path: Path to file (relative to sandbox root)
            require_confirmation: Whether to require user confirmation

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            full_path = self.sandbox_root / file_path

            if not self._is_safe_path(full_path):
                return False

            if not full_path.exists():
                self.logger.warning(f"File not found for deletion: {full_path}")
                return False

            # Security check - always require confirmation for deletions
            if require_confirmation:
                self.logger.warning(f"Deleting file: {full_path}")
                # In a real implementation, this would prompt the user
                # Return False without confirmation in production

            # Delete file or directory
            if full_path.is_file():
                full_path.unlink()
            elif full_path.is_dir():
                full_path.rmdir()  # Only removes empty directories

            self.logger.info(f"Deleted: {full_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {e}")
            return False

    def search_files(self, query: str, directory: Optional[str] = None, 
                    content_search: bool = False) -> List[FileInfo]:
        """
        Search for files by name or content

        Args:
            query: Search query
            directory: Directory to search in (relative to sandbox root)
            content_search: Whether to search file contents (slower)

        Returns:
            List of matching FileInfo objects
        """
        if not self.enabled:
            return []

        try:
            if directory:
                search_dir = self.sandbox_root / directory
            else:
                search_dir = self.sandbox_root

            if not self._is_safe_path(search_dir) or not search_dir.exists():
                return []

            matching_files = []
            query_lower = query.lower()

            for path in search_dir.rglob("*"):
                if not path.is_file() or not self._is_allowed_extension(path):
                    continue

                # Search by filename
                if query_lower in path.name.lower():
                    try:
                        file_info = self._get_file_info(path)
                        matching_files.append(file_info)
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to get info for {path}: {e}")

                # Search by content if requested
                if content_search and path.stat().st_size <= self.max_file_size:
                    try:
                        content = self.read_file(str(path.relative_to(self.sandbox_root)), 
                                               require_confirmation=False)
                        if content and query_lower in content.lower():
                            file_info = self._get_file_info(path)
                            matching_files.append(file_info)
                    except Exception as e:
                        self.logger.warning(f"Failed to search content of {path}: {e}")

            self.logger.info(f"Found {len(matching_files)} files matching query: {query}")
            return matching_files

        except Exception as e:
            self.logger.error(f"Failed to search files: {e}")
            return []

    def get_file_summary(self, file_path: str) -> Optional[Dict]:
        """
        Get summary of file (size, type, basic stats)

        Args:
            file_path: Path to file (relative to sandbox root)

        Returns:
            Dictionary with file summary or None if failed
        """
        try:
            full_path = self.sandbox_root / file_path

            if not self._is_safe_path(full_path) or not full_path.exists():
                return None

            file_info = self._get_file_info(full_path)

            summary = {
                "name": file_info.name,
                "path": file_info.path,
                "size_bytes": file_info.size,
                "size_human": self._format_size(file_info.size),
                "mime_type": file_info.mime_type,
                "last_modified": file_info.last_modified.isoformat(),
                "hash": file_info.hash
            }

            # Add content preview for text files
            if file_info.mime_type.startswith('text/') and file_info.size < 10000:
                content = self.read_file(file_path, require_confirmation=False)
                if content:
                    summary["content_preview"] = content[:500] + "..." if len(content) > 500 else content
                    summary["line_count"] = content.count('\n') + 1
                    summary["word_count"] = len(content.split())


            return summary

        except Exception as e:
            self.logger.error(f"Failed to get file summary for {file_path}: {e}")
            return None

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def get_sandbox_info(self) -> Dict:
        """Get information about the file sandbox"""
        try:
            total_files = len(list(self.sandbox_root.rglob("*")))
            total_size = sum(f.stat().st_size for f in self.sandbox_root.rglob("*") if f.is_file())

            return {
                "sandbox_root": str(self.sandbox_root),
                "enabled": self.enabled,
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_human": self._format_size(total_size),
                "allowed_extensions": list(self.allowed_extensions),
                "max_file_size": self._format_size(self.max_file_size)
            }
        except Exception as e:
            self.logger.error(f"Failed to get sandbox info: {e}")
            return {"error": str(e)}