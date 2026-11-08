9. MTEB Arena: how often a judge picks the side a person voted for, on the Arena's retrieval battles
(mteb/arena-results, retrieval_battle: 4,087 votes; one top document per side, as people saw it).

Grades are in data/arena/<judge>.jsonl, made by arena_judge.py against a served decision model: per battle, a
pointwise grade for each side (ga, gb, the expected grade 0-5) and a pairwise preference for side A judged in both
presentation orders (pa). Agreement counts only decisive votes (A or B) where the two sides differ; tie and
"both bad" votes are reported apart. 95% CI: 2,000 bootstrap resamples of the battles, seed 0.

| judge | judging | battles | decisive | scored | agreement | ci_low | ci_high | votes |
|---|---|---|---|---|---|---|---|---|
| JEV-27B | pointwise | 4085 | 2169 | 2169 | 0.812 | 0.794 | 0.828 | A=1152 B=1039 both_bad=699 tie=1195 |
| JEV-27B | pairwise | 4085 | 2169 | 2169 | 0.820 | 0.804 | 0.837 | A=1152 B=1039 both_bad=699 tie=1195 |
| AutoTrust-JEV-27B-v1.1 | pointwise | 4085 | 2169 | 2169 | 0.845 | 0.829 | 0.859 | A=1152 B=1039 both_bad=699 tie=1195 |
| AutoTrust-JEV-27B-v1.1 | pairwise | 4085 | 2169 | 2168 | 0.845 | 0.830 | 0.861 | A=1152 B=1039 both_bad=699 tie=1195 |
| JEV-35B | pointwise | 4085 | 2169 | 2169 | 0.834 | 0.817 | 0.850 | A=1152 B=1039 both_bad=699 tie=1195 |
| JEV-35B | pairwise | 4085 | 2169 | 2169 | 0.836 | 0.820 | 0.852 | A=1152 B=1039 both_bad=699 tie=1195 |
