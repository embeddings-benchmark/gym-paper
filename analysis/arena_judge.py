"""Judge the MTEB Arena retrieval battles with a served decision model, for arena.py.

Each side is the top document the person saw. For each battle the judge grades both sides (ga, gb: the expected grade,
0-5) and says which side it prefers, asked in both orders (pa: the chance it prefers A). Results are appended to the
output file, so a stopped run picks up where it left off.

    python arena_judge.py ../data/arena/<judge>.jsonl http://host:port [http://host:port ...]
"""

import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import arena_battles as battles

GRADE = ("How relevant is the document to the query? 0 = not relevant at all, 5 = perfectly relevant: "
         "it fully satisfies the information need.")
PREFER = ("Which result list better satisfies the query, judging relevance, coverage of the information need and "
          "ranking quality?")
OPTIONS = ["System A", "System B", "Tie"]


def cut(text, n):
    """Cut text to n characters at a word boundary."""
    if len(text) <= n:
        return text
    head = text[:n].rsplit(None, 1)
    return (head[0] if len(head) > 1 else text[:n]) + "..."


servers = itertools.cycle(sys.argv[2:])
lock = threading.Lock()


def decide(body):
    for attempt in range(8):
        with lock:
            url = next(servers)
        try:
            r = requests.post(url + "/v1/decide", json=body, headers={"Authorization": "Bearer gym"}, timeout=900)
            r.raise_for_status()
            out = r.json()
            return dict(zip(out["options"], out["probabilities"]))
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("decide failed")


def grade(query, doc):
    p = decide({"kind": "score", "state": f"Query: {query}\nDocument: {cut(doc, 16000)}", "question": GRADE})
    return sum(int(k) * v for k, v in p.items())


def prefer_first(query, first, second):
    state = f"Query: {query}\n\nSystem A results:\n  1. {cut(first, 4000)}\n\nSystem B results:\n  1. {cut(second, 4000)}"
    return decide({"kind": "choice", "state": state, "question": PREFER, "options": OPTIONS})


def judge(b):
    if b["id"] in done:
        return
    same = b["a"] == b["b"]
    pa = 0.5
    if not same:
        ab, ba = prefer_first(b["query"], b["a"], b["b"]), prefer_first(b["query"], b["b"], b["a"])
        pa = ((ab["System A"] + 0.5 * ab["Tie"]) + (ba["System B"] + 0.5 * ba["Tie"])) / 2
    row = {"id": b["id"], "vote": b["vote"], "ga": grade(b["query"], b["a"]), "gb": grade(b["query"], b["b"]), "pa": pa,
           "ma": b["ma"], "mb": b["mb"], "corpus": b["corpus"], "same": same}
    with lock:
        out.write(json.dumps(row) + "\n")
        out.flush()


try:
    done = {json.loads(line)["id"] for line in open(sys.argv[1])}
except FileNotFoundError:
    done = set()
out = open(sys.argv[1], "a")
with ThreadPoolExecutor(96) as pool:
    list(pool.map(judge, battles()))
