"""In-memory registry for unique dataset manifest versions."""

from project_genesis.datasets.manifest import DatasetManifest


class DatasetRegistry:
    """Register and retrieve manifests by dataset name and version."""

    def __init__(self) -> None:
        """Create an empty in-memory registry."""
        self._manifests: dict[tuple[str, str], DatasetManifest] = {}

    def register(self, manifest: DatasetManifest) -> None:
        """Register a manifest, rejecting ambiguous duplicate versions."""
        key = (manifest.metadata.name, manifest.metadata.version)
        if key in self._manifests:
            raise ValueError(f"dataset is already registered: {key[0]} {key[1]}")
        self._manifests[key] = manifest

    def get(self, name: str, version: str) -> DatasetManifest:
        """Return one registered manifest."""
        try:
            return self._manifests[(name, version)]
        except KeyError as error:
            raise KeyError(f"dataset is not registered: {name} {version}") from error

    def versions(self, name: str) -> tuple[str, ...]:
        """Return sorted registered versions for a dataset name."""
        return tuple(sorted(version for dataset, version in self._manifests if dataset == name))

    def __len__(self) -> int:
        """Return the number of registered dataset versions."""
        return len(self._manifests)
