#!/usr/bin/env python3
"""Validate that the iOS project is local, dependency-free, and credential-free."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_PROJECT_MARKERS = (
    "XCRemoteSwiftPackageReference",
    "XCSwiftPackageProductDependency",
    "repositoryURL",
    "Alamofire",
    "ChatGPTSwift",
)
FORBIDDEN_IMPORT = re.compile(r"^\s*import\s+(Alamofire|ChatGPTSwift)\b", re.MULTILINE)
SECRET_PATTERNS = (
    re.compile(r"\bksk_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[\"'][^\"']{8,}[\"']"),
)
FILE_REFERENCE = re.compile(
    r"isa\s*=\s*PBXFileReference;[^}]*?path\s*=\s*(?:\"([^\"]+)\"|([^;\s]+))\s*;",
    re.DOTALL,
)
SOURCE_OR_RESOURCE = {".swift", ".json", ".xcassets"}


def validate_project(project_file: Path) -> list[str]:
    errors: list[str] = []
    if not project_file.is_file():
        return [f"project file does not exist: {project_file}"]

    project_root = project_file.parent.parent.resolve()
    text = project_file.read_text(encoding="utf-8")
    for marker in FORBIDDEN_PROJECT_MARKERS:
        if marker in text:
            errors.append(f"forbidden package marker: {marker}")

    referenced_names: set[str] = set()
    for match in FILE_REFERENCE.finditer(text):
        raw_path = match.group(1) or match.group(2)
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"external file reference: {raw_path}")
            continue
        if path.suffix not in SOURCE_OR_RESOURCE:
            continue
        referenced_names.add(path.name)
        candidates = list(project_root.rglob(path.name))
        if not candidates:
            errors.append(f"missing file reference: {raw_path}")
        elif any(not candidate.resolve().is_relative_to(project_root) for candidate in candidates):
            errors.append(f"reference escapes project root: {raw_path}")

    expected = {
        "AcharyaGPT_iOS_App.swift",
        "ChatbotView.swift",
        "ChatModels.swift",
        "ChatAPIClient.swift",
        "ChatViewModel.swift",
        "SupportingViews.swift",
        "ChatClientTests.swift",
    }
    for missing in sorted(expected - referenced_names):
        errors.append(f"required source is not referenced: {missing}")

    package_lock = project_file.parent / "project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
    if package_lock.exists():
        errors.append(f"stale package lock exists: {package_lock}")

    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".swift", ".json", ".plist"}:
            continue
        contents = path.read_text(encoding="utf-8", errors="replace")
        if FORBIDDEN_IMPORT.search(contents):
            errors.append(f"forbidden external import in {path.relative_to(project_root)}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(contents):
                errors.append(f"secret-like value in {path.relative_to(project_root)}")
                break

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append("secret-like value in project build settings")
            break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_project(args.project.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Xcode project structural validation: PASSED")
    print("macOS Xcode execution: PENDING_MACOS_VALIDATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
