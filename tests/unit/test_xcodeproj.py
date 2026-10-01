from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _validator(project_root: Path):  # type: ignore[no-untyped-def]
    path = project_root / "scripts/validate_xcodeproj.py"
    spec = importlib.util.spec_from_file_location("validate_xcodeproj", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ios_project_has_only_local_secret_free_references(project_root: Path) -> None:
    validator = _validator(project_root)
    project = project_root / "AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj/project.pbxproj"
    assert validator.validate_project(project) == []


def test_validator_rejects_remote_packages(project_root: Path, tmp_path: Path) -> None:
    validator = _validator(project_root)
    source = project_root / "AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj/project.pbxproj"
    project_dir = tmp_path / "App/App.xcodeproj"
    project_dir.mkdir(parents=True)
    project = project_dir / "project.pbxproj"
    project.write_text(source.read_text() + "\nXCRemoteSwiftPackageReference\n")
    errors = validator.validate_project(project)
    assert any("forbidden package marker" in error for error in errors)


@pytest.mark.parametrize("secret", ["ksk_ABCDEFGHIJKLMNOPQRST", "sk-ABCDEFGHIJKLMNOPQRST"])
def test_validator_rejects_secret_like_client_values(
    project_root: Path, tmp_path: Path, secret: str
) -> None:
    validator = _validator(project_root)
    project_root_copy = tmp_path / "App"
    source_root = project_root_copy / "Sources"
    source_root.mkdir(parents=True)
    (source_root / "ChatAPIClient.swift").write_text(f'let credential = "{secret}"')
    project_dir = project_root_copy / "App.xcodeproj"
    project_dir.mkdir()
    project = project_dir / "project.pbxproj"
    project.write_text(
        "isa = PBXFileReference; lastKnownFileType = sourcecode.swift; "
        "path = ChatAPIClient.swift; sourceTree = \"<group>\";"
    )
    errors = validator.validate_project(project)
    assert any("secret-like value" in error for error in errors)
