# Moral Stories Dataset Investigation

## Plan:
- [ ] List all files and directories in the root directory.
- [ ] Identify data files (.jsonl, .csv, .parquet, .json).
- [ ] Explore likely subdirectories for data files.
- [ ] Summarize the exact paths to data files.

## Findings:
- **Root**:
  - `data/` (Directory)
  - `.gitattributes`
  - `README.md`
  - `dataset_infos.json` (Contains metadata)
  - `moral_stories.py` (HF dataset script)
- **`data/` Directory**:
  - `classification/` (Directory)
  - `generation/` (Directory)
  - `.gitattributes`
  - `moral_stories_full.jsonl` (Full dataset, 8.02 MB)
- **`data/classification/`**:
  - Task folders: `action+context+consequence`, `action+context`, `action+norm`, `action`, `consequence+action+context`, `consequence+action`
  - Each task folder contains split folders: `lexical_bias`, `minimal_pairs`, `norm_distance`
  - Each split folder contains: `train.jsonl`, `test.jsonl`, `valid.jsonl`
- **`data/generation/`**:
  - Task folders: `action|context+consequence`, `action|context`, `consequence|action+context`, `consequence|action`, `norm|actions+context+consequences`, `norm|actions+context`, `norm|actions`
  - Each task folder contains split folders: `lexical_bias`, `minimal_pairs`, `norm_distance`
  - Each split folder contains: `train.jsonl`, `test.jsonl`, `valid.jsonl`
