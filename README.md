# MTEB Gym paper

The analyses behind the MTEB Gym paper. Every number is computed from published data:

- **Records:** [embeddings-benchmark/gym-results](https://github.com/embeddings-benchmark/gym-results), 588 records: the main table (JEV-27B, pointwise and pairwise, original and DeepSeek-V4-Flash queries, 28 tasks), plus the other judges and generators.
- **What the records were computed from:** [mteb/gym-runs](https://huggingface.co/datasets/mteb/gym-runs), with the queries, predictions, grades and verdicts.
- **What isn't published elsewhere:** in `data/` here (see below).

The library is [MTEB-gym](https://github.com/embeddings-benchmark/MTEB-gym). This repo only reads its results; nothing here changes a record.

Layout: `analysis/` holds the scripts that make each table, `out/` their outputs, and `data/` the small inputs. The LaTeX goes in `paper/`.

## Run

```bash
uv venv -p 3.12 && uv pip install -r pyproject.toml
cd analysis
python 1_main_table.py   # each script writes its tables to ../out/ as CSV and Markdown
```

Records are fetched into the cache, as `mteb_gym.load_results()` fetches them. To use a local results folder instead, set `GYM_RESULTS=/path/to/results`. Scripts 7 and `labels.py` also download from `mteb/gym-runs`: about 2 GB of grades and predictions.

## Scripts

| Script | Makes | Paper issue |
|---|---|---|
| `1_main_table.py` | Agreement with labels (Spearman, Kendall) per judging mode and query source, with bootstrap CIs over tasks | #47 |
| `2_judges_generators.py` | Every other judge and generator against JEV-27B: paired difference, CI, and the .03 rule fixed before results | #46 |
| `3_top_weighted.py` | Rank-biased overlap at p = 0.9 (Webber et al., 2010), top-1, and top pick in the labels' top 3 | #47 |
| `4_regret.py` | nDCG@10 points lost by picking each method's top model | #47 |
| `5_leaderboard_baseline.py` | Agreement of rankings from leaderboard history alone, and each method's paired difference from it | #47 |
| `6_unseen_models.py` | Mean rank error when placing one model among the others, by method, history, or nothing | #47 |
| `7_query_count.py` | Agreement and regret with 10, 25, 50 and all queries (20 draws per task), rebuilt from the published grades | #52 |
| `8_cost.py` | Measured speeds and token counts from the run logs | |
| `9_arena.py` | How often a judge picks the side a person voted for, on MTEB Arena's 4,087 retrieval battles | #50 |
| `labels.py` | Rebuilds `data/labels` from the published predictions, and checks the files here against it | |
| `arena_judge.py` | Judges the Arena battles with a served decision model, writing `data/arena/<judge>.jsonl` | #50 |

Shared code: `common.py` (records, labels, bootstrap, tables) and `history.py` (the two leaderboard-history baselines).

## Data

| Path | What | How it was made |
|---|---|---|
| `data/tasks.txt` | The 28 tasks (#44) | |
| `data/labels/<task>.json` | Each model's nDCG@10 against the task's own labels, on the 100 original queries the gym sampled | `labels.py --write`, from the published predictions and the task's qrels, with mteb's metric code. The three BRIGHT tasks with excluded documents drop them, as mteb does |
| `data/published_scores.json` | Each model's published MTEB score on each task | `history.py`, once, from mteb's result cache |
| `data/low_effort_*.tsv` | The chat judges' grades that fell back to low reasoning effort | Logged during the judge runs |
| `data/arena/<judge>.jsonl` | Arena grades for JEV-27B, AutoTrust-JEV-27B-v1.1 and JEV-35B | `arena_judge.py`, on 2026-11-06 |
| `data/logs/` | Lines from the run logs that `8_cost.py` reads: timestamps, grade rates, token counts | |

## Choices

- **Bootstrap:** 10,000 resamples of the 28 tasks, seed 0. Script 6 resamples placements, which are not independent within a task. The Arena uses 2,000 resamples of battles.
- **History baselines:** "history (labels)" ranks a model by its mean rank on the other 27 tasks' labels; "history (published)" does the same with the published MTEB scores, counting each task only for the models that have one. This is what a practitioner would do without the gym: pick the model that does best elsewhere.
- **Arena agreement:** counts decisive votes (A or B) where the two sides differ. Tie and "both bad" votes are reported apart. Each battle is one document per side, as people saw it, so the Arena tests the judge, not the gym's ranking.
- **Script 7:** checks its rebuilt per-query scores against each record's ratings, and its label nDCG against `data/labels`. See `out/7_query_count_check.*`.
