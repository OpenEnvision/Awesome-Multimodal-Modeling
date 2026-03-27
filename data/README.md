# Awesome README Data

This folder is the source of truth for `README.md`.

## Files

- `meta.json`: hero badges, navigation buttons, and footer metadata
- `content.json`: section hierarchy and prose blocks
- `tables.json`: structured tables for models, datasets, benchmarks, and resources

## Workflow

1. Edit `data/meta.json`, `data/content.json`, or `data/tables.json`
2. Regenerate the README:

```bash
python3 scripts/generate_awesome_readme.py \
  --meta data/meta.json \
  --content data/content.json \
  --tables data/tables.json \
  --output README.md
```

## Bootstrap / Migration

If you ever need to rebuild the JSON data from a hand-edited README, use:

```bash
python3 scripts/migrate_awesome_readme_to_data.py \
  --input README.md \
  --content-output data/content.json \
  --tables-output data/tables.json
```

That migration script is intended for bootstrapping, not day-to-day editing.
