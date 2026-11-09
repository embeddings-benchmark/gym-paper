"""Simple ways to pick a side on MTEB Arena's retrieval battles, to compare with the LLM judges (for plot_arena.py).

For each battle, a score for side A and side B under each method; the method picks the side with the higher score:
- length: the longer document;
- overlap: the share of the query's words that appear in the document;
- an embedding model as the judge: cosine similarity between query and document (sentence-transformers, with the
  model's query prompt where it has one).

Writes data/arena/baselines.jsonl: {id, vote, same, <method>: [score A, score B], ...}.

    python arena_baselines.py
"""

import json
import re

from sentence_transformers import SentenceTransformer

from common import DATA, arena_battles

EMBEDDERS = {"all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", None),
             "Qwen3-Embedding-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "query")}

battles = list({b["id"]: b for b in arena_battles()}.values())
words = lambda text: set(re.findall(r"\w+", text.lower()))
rows = []
for b in battles:
    q = words(b["query"])
    overlap = [len(q & words(b[s])) / max(len(q), 1) for s in "ab"]
    rows.append({"id": b["id"], "vote": b["vote"], "same": b["a"] == b["b"], "length": [len(b["a"]), len(b["b"])],
                 "overlap": overlap})

for name, (path, prompt) in EMBEDDERS.items():
    model = SentenceTransformer(path, device="cuda")
    model.max_seq_length = min(model.max_seq_length, 512)
    queries = model.encode([b["query"] for b in battles], prompt_name=prompt, normalize_embeddings=True, batch_size=64)
    for side in "ab":
        docs = model.encode([b[side] for b in battles], normalize_embeddings=True, batch_size=64)
        for row, qv, dv in zip(rows, queries, docs):
            row.setdefault(name, []).append(float(qv @ dv))
    print(name, "done", flush=True)

with open(DATA / "arena" / "baselines.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
print(len(rows), "battles")
