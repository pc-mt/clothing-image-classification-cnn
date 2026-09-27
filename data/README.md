# Dataset

The image dataset is not committed to this repository. This keeps the Git history small and avoids redistributing data without a documented licence.

Expected layout after preparation:

```text
data/clothing_clean_5classes_split_audit_copy/
├── train/{long-sleeved,pants,shorts,socks,t-shirt}/
├── val/{long-sleeved,pants,shorts,socks,t-shirt}/
└── test/{long-sleeved,pants,shorts,socks,t-shirt}/
```

Prepare a local copy with:

```bash
python scripts/prepare_dataset.py /path/to/clothing_clean_5classes_split_audit_copy.zip
```

The preparation script preserves the source archive, extracts the images and removes exact cross-split duplicates from the extracted copy.
