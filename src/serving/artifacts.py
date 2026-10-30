"""Versioned artifact manifest and safe serving-bundle validation."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_path(posix_path: str) -> Path:
    """Unpickle hook for paths written by the bundle builder on any OS."""
    return Path(posix_path)


class ArtifactSpec(BaseModel):
    path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required: bool = True


class ServingManifest(BaseModel):
    schema_version: Literal["1"] = "1"
    model_version: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    item_mapping_version: str = Field(min_length=1)
    num_users: int = Field(ge=0)
    num_items: int = Field(gt=0)
    embedding_dimension: int | None = Field(default=None, gt=0)
    model_type: str | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    artifacts: dict[str, ArtifactSpec]

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "ServingManifest":
        manifest_path = Path(bundle_dir) / "manifest.json"
        with manifest_path.open(encoding="utf-8") as file:
            return cls.model_validate(json.load(file))

    def validate_files(self, bundle_dir: str | Path, verify_checksums: bool = True) -> None:
        """Validate required files and prevent paths from escaping the bundle."""
        root = Path(bundle_dir).resolve()
        for name, spec in self.artifacts.items():
            artifact_path = (root / spec.path).resolve()
            if artifact_path != root and root not in artifact_path.parents:
                raise ValueError(f"artifact '{name}' escapes bundle directory")
            if not artifact_path.is_file():
                if spec.required:
                    raise FileNotFoundError(
                        f"required artifact '{name}' not found: {artifact_path}"
                    )
                continue
            if verify_checksums and spec.sha256:
                if sha256_file(artifact_path) != spec.sha256:
                    raise ValueError(f"checksum mismatch for artifact '{name}'")
