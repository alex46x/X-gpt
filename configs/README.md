# Configuration

Configuration is versioned YAML grouped by its owning subsystem:

```text
configs/
├── model/
├── tokenizer/
├── dataset/
├── training/
├── optimizer/
├── inference/
├── evaluation/
├── logging/
└── experiment/
```

Only `dataset/default.yaml` exists in Phase 2 because it is the only implemented
consumer. Other directories and defaults are added with their subsystem rather
than carrying unvalidated placeholder settings.

Configuration rules:

- Use plain YAML mappings, lists, and scalar values without custom tags.
- Keep values external; Python owns schema and validation, not experiment values.
- Resolve relative paths from the YAML file that declares them.
- Reject unknown fields.
- Use dotted `key=value` overrides only for keys already present in the file.
- Set `PROJECT_GENESIS_ENV` to `development`, `test`, or `production` when the
  process environment must override the file.

These rules preserve a direct path to Hydra/OmegaConf composition later without
depending on either tool before composition is needed.
