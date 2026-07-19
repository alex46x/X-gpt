# Phase 3: Data Cleaning and Preprocessing

## Goal

Phase 3 converts verified local source files into normalized immutable
`DatasetRecord` values. It produces quality statistics and a processed-data
manifest that binds raw inputs, reader settings, preprocessing policy, and output
records.

No tokenizer, vocabulary, tensor, model, or training behavior is present.

## Architecture

```text
configuration
    ↓
datasets
    ↓
preprocessing.config
    ↓
preprocessing.readers → preprocessing.pipeline
                              ├── Dataset
                              ├── QualityReport
                              └── ProcessedDatasetManifest
```

The implementation is split by responsibility:

- `config.py` validates external normalization, filtering, deduplication, and
  error policy.
- `readers.py` decodes local source formats into `RawDocument` values.
- `pipeline.py` verifies raw files, normalizes and filters text, performs exact
  deduplication, creates records, and emits reports and manifests.

## Supported local sources

- Plain text and Markdown produce one document per file.
- JSON accepts a string, an object with the configured text field, or a list of
  those values.
- JSONL produces one document per non-empty line.
- CSV produces one document per row using the configured text column.
- PDF produces one document per page through PyPDF.
- Git snapshots read deterministically selected local files and exclude `.git`
  internals.
- Web snapshots extract visible text from local HTML while ignoring script and
  style content.

Readers process files in normalized path order. JSON arrays, JSONL lines, CSV
rows, and PDF pages preserve their source order. File decoding is explicit and
defaults to UTF-8.

## Normalization and filtering

The versioned YAML policy controls Unicode normalization, newline normalization,
control-character removal, whitespace collapsing, trimming, minimum and maximum
character counts, allowed languages, exact deduplication, and parse-error
behavior.

Defaults preserve line structure because Markdown and source code are future
training inputs. Whitespace collapsing is available but disabled. Cleaning does
not tokenize or infer language, licenses, labels, or quality scores.

## Identity and reproducibility

Each output record has a stable identifier derived from its normalized relative
source path and source position. The record checksum covers cleaned text, while
`raw_checksum` preserves the exact parsed-text identity.

`ProcessedDatasetManifest` records:

- Raw dataset manifest fingerprint.
- Reader and preprocessing configuration fingerprint.
- Output dataset fingerprint.
- Accepted, filtered, duplicate, and parser-failure counts.
- Character counts and rejection reasons.

The manifest is written atomically as canonical, versioned JSON. Creation clocks
and absolute machine paths do not affect fingerprints.

## Failure behavior and limits

Raw manifest corruption always stops processing. `on_error: raise` stops on the
first parser failure; `on_error: skip` records the failure and continues with the
next file. Unknown configuration keys, invalid encodings, symlinks, nested JSON
metadata, non-finite JSON numbers, and missing text fields fail explicitly.

Exact deduplication currently holds SHA-256 checksums in memory. This is simple
and deterministic; a partitioned disk-backed set becomes necessary only when
checksum memory is measured as material at corpus scale.

PDF extraction quality depends on embedded text. OCR, language detection,
near-duplicate detection, distributed execution, and semantic quality models are
not implemented because they require separate measured requirements and
dependencies.
