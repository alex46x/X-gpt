# Phase 2: Configuration Contracts and Dataset Foundation

## Architecture

Phase 2 adds two packages:

```text
src/project_genesis/
├── configuration/
│   ├── loader.py       Safe YAML loading and dotted overrides
│   └── models.py       Runtime environment and shared path models
└── datasets/
    ├── config.py       Dataset-owned typed configuration
    ├── integrity.py    Streaming SHA-256 primitives
    ├── manifest.py     Deterministic file inventory and verification
    ├── records.py      Immutable records, schema, collection, and statistics
    ├── registry.py     Unique name/version manifest registry
    ├── sources.py      Local source, format, and split contracts
    └── storage.py      Manifest storage boundary and atomic local backend
```

The configuration package has no dataset dependency. The dataset package consumes
configuration primitives and owns the only typed subsystem schema currently
implemented.

## Configuration decisions

- **PyYAML safe loading:** YAML is required by the project. `safe_load` rejects
  object construction tags, and normalized values are limited to mappings,
  lists, and scalar values.
- **Typed schemas at consumers:** a generic untyped application configuration
  would hide misspellings. `DatasetConfig` therefore validates all keys and
  converts values into immutable models.
- **Strict overrides:** dotted assignments are compatible with future CLI
  arguments but may only update existing keys. An override cannot silently add
  a misspelled setting.
- **Path ownership:** relative paths resolve from the declaring YAML file, making
  execution independent of the current working directory.
- **Environment precedence:** `PROJECT_GENESIS_ENV` overrides the file; the file
  overrides the development default.
- **Hydra compatibility:** configuration uses ordinary nested YAML without
  custom tags or interpolation. Hydra/OmegaConf remains an integration option
  when multi-file experiment composition justifies it.

## Dataset decisions

- **Immutable records:** frozen, slotted dataclasses protect document identity,
  provenance, timestamps, checksums, and metadata after validation.
- **Reserved future fields:** optional immutable token IDs, embeddings, and
  labels define compatibility without creating or interpreting those values.
- **Content identity:** record and file checksums use SHA-256. Dataset and
  manifest fingerprints use canonical JSON and exclude machine-specific paths
  and creation times.
- **Deterministic local inventory:** source files are sorted by normalized
  relative path. Symlinks and files outside the configured root are rejected.
- **Integrity versus quality:** missing or modified files invalidate a manifest.
  Duplicate checksums are reported separately because duplicates are a cleaning
  decision for Phase 3, not corruption.
- **Minimal extension boundaries:** `ManifestStorage` supports alternate storage
  later. `DatasetCache` uses the standard mutable-mapping contract, so a normal
  dictionary works until measured cache requirements justify a backend.
- **No parsers:** source formats declare text, Markdown, JSON, JSONL, CSV, PDF,
  Git snapshots, and web snapshots. Format parsing and cleaning belong to Phase
  3 and are deliberately absent.

## Validation and failure behavior

Configuration rejects missing files, non-YAML extensions, unsafe YAML, non-mapping
roots, unsupported value types, unknown fields, missing fields, invalid
environments, invalid semantic versions, malformed timestamps, nonexistent
sources, and unsafe paths.

Records reject empty required strings, naive timestamps, malformed or mismatched
checksums, mutable or unsupported metadata, negative token IDs or labels, and
non-finite or ragged embeddings. Datasets reject duplicate document IDs.

Manifests reject unsorted or duplicate paths, path traversal, symlinks, sources
outside the root, overlapping source declarations, checksum tampering, and
invalid persisted fingerprints. Local storage writes manifests atomically.

## Future compatibility

Phase 3 can consume `DatasetSource` declarations and emit `DatasetRecord` values
without changing the contracts. Later tokenizer and embedding phases may populate
the reserved optional fields or introduce purpose-built derived-record types if
measurements show that storing large tuples on source records is unsuitable.
Remote storage can implement `ManifestStorage`; distributed registries and caches
are deferred until there is a real multi-process consumer.
