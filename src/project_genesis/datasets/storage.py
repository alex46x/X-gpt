"""Atomic local persistence for dataset manifests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from project_genesis.datasets.manifest import DatasetManifest
from project_genesis.utilities import atomic_write_text


class ManifestStorage(Protocol):
    """Storage boundary for local or future remote manifest backends."""

    def save(self, manifest: DatasetManifest) -> Path:
        """Persist a manifest and return its storage location."""
        ...

    def load(self, name: str, version: str) -> DatasetManifest:
        """Load a manifest by dataset identity."""
        ...


@dataclass(frozen=True, slots=True)
class LocalManifestStorage:
    """Store manifests atomically beneath one local directory."""

    root: Path

    def save(self, manifest: DatasetManifest) -> Path:
        """Atomically persist a manifest as UTF-8 JSON."""
        destination = self._path(manifest.metadata.name, manifest.metadata.version)
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        atomic_write_text(destination, payload)
        return destination

    def load(self, name: str, version: str) -> DatasetManifest:
        """Load and validate a stored manifest."""
        path = self._path(name, version)
        try:
            with path.open(encoding="utf-8") as stream:
                decoded: object = json.load(stream)
            return DatasetManifest.from_dict(decoded)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Unable to load manifest {name} {version}: {error}") from error

    def _path(self, name: str, version: str) -> Path:
        for label, value in (("name", name), ("version", version)):
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"dataset {label} is not a safe path component: {value!r}")
        return self.root.expanduser().resolve() / name / version / "manifest.json"
