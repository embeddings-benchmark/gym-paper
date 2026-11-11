"""Appendix tables listing the 28 tasks and the 28 models, with their citations.

    tables/tasks.tex         each task: its source, corpus size, the task's own queries, the queries the gym used
                             with original queries, and the mean pool size with generated queries
    tables/instructions.tex  each task's instruction
    tables/models.tex   each model: parameters, embedding size and maximum input length, from mteb

Corpus and query counts are in data/task_sizes.json, counted with mteb_gym.corpus.load on the corpus each record names
(`python appendix_lists.py --count` recounts them). Citations were checked against each paper's own record (ACL
Anthology, DBLP, arXiv, the publisher); the sources are listed in data/citations.md.
BEIR, which most of the MTEB(eng, v2) tasks come through, is cited in the text."""

import json
import sys

import mteb

from common import DATA, GENERATOR, JUDGE, TABLES, TASKS, labels, records
from plots import NAMES

# The paper that introduced each dataset; HardNegatives versions are MMTEB's, Touché v3 is Thakur et al.'s.
TASK_CITES = {
    "ArguAna": "wachsmuth2018arguana",
    "ClimateFEVERHardNegatives": "diggelmann2020climatefever,enevoldsen2025mmteb",
    "CQADupstackGamingRetrieval": "hoogeveen2015cqadupstack",
    "CQADupstackUnixRetrieval": "hoogeveen2015cqadupstack",
    "FEVERHardNegatives": "thorne2018fever,enevoldsen2025mmteb",
    "FiQA2018": "maia2018fiqa",
    "HotpotQAHardNegatives": "yang2018hotpotqa,enevoldsen2025mmteb",
    "SCIDOCS": "cohan2020specter",
    "Touche2020Retrieval.v3": "bondarenko2020touche,thakur2024touche",
    "TRECCOVID": "voorhees2020treccovid,roberts2021treccovid",
    "AILACasedocs": "bhattacharya2019aila,bhattacharya2020ailadata",
    "AppsRetrieval": "hendrycks2021apps,li2025coir",
    "CUREv1": "sheikh2025cure",
    "FinQARetrieval": "chen2021finqa",
    "NFCorpus": "boteva2016nfcorpus",
    "SciFact": "wadden2020scifact",
}
TASK_CITES.update({t: "su2024bright" for t in TASKS if t.startswith("Bright")})
# The paper or model card for each model.
MODEL_CITES = {
    "Alibaba-NLP/gte-Qwen2-7B-instruct": "li2023gte",
    "BAAI/bge-base-en-v1.5": "xiao2024cpack",
    "BAAI/bge-large-en-v1.5": "xiao2024cpack",
    "BAAI/bge-small-en-v1.5": "xiao2024cpack",
    "IEITYuan/Yuan-embedding-2.0-en": "ieityuan2025yuanembedding",
    "Linq-AI-Research/Linq-Embed-Mistral": "choi2024linq",
    "Qwen/Qwen3-Embedding-0.6B": "zhang2025qwen3embedding",
    "Qwen/Qwen3-Embedding-4B": "zhang2025qwen3embedding",
    "Qwen/Qwen3-Embedding-8B": "zhang2025qwen3embedding",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "yu2024arcticembed2",
    "codefuse-ai/F2LLM-v2-4B": "zhang2026f2llmv2",
    "colbert-ir/colbertv2.0": "santhanam2022colbertv2",
    "google/embeddinggemma-300m": "vera2025embeddinggemma",
    "ibm-granite/granite-embedding-english-r2": "awasthy2025granite",
    "intfloat/e5-mistral-7b-instruct": "wang2024e5mistral",
    "intfloat/multilingual-e5-large-instruct": "wang2024multilinguale5",
    "jinaai/jina-embeddings-v5-text-nano": "akram2026jinav5",
    "jinaai/jina-embeddings-v5-text-small": "akram2026jinav5",
    "lightonai/DenseOn": "sourty2026denseon",
    "microsoft/harrier-oss-v1-0.6b": "microsoft2026harrier",
    "minishlab/potion-retrieval-32M": "tulkens2024model2vec",
    "mixedbread-ai/mxbai-embed-large-v1": "lee2024mxbai,li2024angle",
    "mteb/baseline-bm25s": "lu2024bm25s",
    "nomic-ai/nomic-embed-text-v1.5": "nussbaum2025nomic",
    "nomic-ai/nomic-embed-text-v2-moe": "nussbaum2025moe",
    "sentence-transformers/all-MiniLM-L6-v2": "reimers2019sbert,wang2020minilm",
    "thenlper/gte-large": "li2023gte",
    "thenlper/gte-small": "li2023gte",
}

OTHER = ["SciFact", "NFCorpus", "FinQARetrieval", "AppsRetrieval", "AILACasedocs", "CUREv1"]  # as in appendix_tables
GROUPS = [("MTEB(eng, v2)", [t for t in TASKS if not t.startswith("Bright") and t not in OTHER]),
          ("BRIGHT", [t for t in TASKS if t.startswith("Bright")]),
          ("Other domains", OTHER)]

def count():
    from mteb_gym.corpus import load

    sizes = {}
    for t in TASKS:
        c = load(t)
        sizes[t] = {"documents": len(c.docs), "queries": len(c.queries or {}), "corpus": c.id}
        print(t, sizes[t])
    (DATA / "task_sizes.json").write_text(json.dumps(sizes, indent=1))


def cite(cites, key):
    return rf"\citealp{{{cites[key]}}}" if key in cites else ""


def tasks_table():
    from mteb_gym.task_instructions import TASK_INSTRUCTIONS

    sizes = json.loads((DATA / "task_sizes.json").read_text())
    R = records()
    original, generated = R[(JUDGE, "pointwise", "original")], R[(JUDGE, "pointwise", GENERATOR)]
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Task} & \textbf{Source} & \textbf{Documents} & \textbf{Queries} & \textbf{Used} & \textbf{Pool} \\"]
    instructions = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{12.4cm}@{}}", r"\toprule",
                    r"\textbf{Task} & \textbf{Task instruction} \\"]
    for group, ts in GROUPS:
        lines += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        instructions += [r"\midrule", rf"\multicolumn{{2}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        for t in ts:
            s = sizes[t]
            used = original[t]["diagnostics"]["queries_rated"]
            pool = generated[t]["diagnostics"]["pool_size_mean"]
            lines.append(rf"{NAMES[t]} & {cite(TASK_CITES, t)} & {s['documents']:,} & {s['queries']:,} & {used} & "
                         rf"{pool:.0f} \\")
            instructions.append(rf"{NAMES[t]} & {TASK_INSTRUCTIONS[t]} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The 28 tasks. \textbf{Documents}: the corpus; \textbf{Queries}: the task's own labeled queries; "
              r"\textbf{Used}: how many of them the gym judged when run on original queries (100, or all if fewer); "
              r"\textbf{Pool}: the mean number of documents judged per generated query, the union of the 28 models' "
              r"top 10.}",
              r"\label{tab:tasks}", r"\end{table*}"]
    instructions += [r"\bottomrule", r"\end{tabular}",
                     r"\caption{The task instruction for each task, given to the query writer and to the judge.}",
                     r"\label{tab:instructions}", r"\end{table*}"]
    (TABLES / "tasks.tex").write_text("\n".join(lines) + "\n")
    (TABLES / "instructions.tex").write_text("\n".join(instructions) + "\n")
    print("wrote", TABLES / "tasks.tex", TABLES / "instructions.tex")


def size(n):
    if not n:
        return "--"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def models_table():
    names = sorted(labels(TASKS[0]))
    metas = {m: mteb.get_model_meta(m) for m in names}
    names.sort(key=lambda m: (metas[m].n_parameters or 0, m.lower()))
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{Source} & \textbf{Params} & \textbf{Dim} & \textbf{Max len} \\", r"\midrule"]
    for m in names:
        meta = metas[m]
        dim = meta.embed_dim
        dim = max(dim) if isinstance(dim, list) else dim
        dim = "--" if dim is None else f"{dim:,}"
        tokens = meta.max_tokens
        tokens = "--" if tokens is None else ("any" if tokens == float("inf") else f"{int(tokens):,}")
        name = m.split("/")[-1].replace("_", r"\_")
        lines.append(rf"{name} & {cite(MODEL_CITES, m)} & {size(meta.n_parameters)} & {dim} & {tokens} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{The 28 models, by size, as mteb describes them: parameters, embedding dimension (the largest, "
              r"for models that offer several) and maximum input length in tokens. ColBERTv2 keeps one vector per "
              r"token; BM25 has no embeddings.}",
              r"\label{tab:models}", r"\end{table*}"]
    (TABLES / "models.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TABLES / "models.tex")


if __name__ == "__main__":
    if "--count" in sys.argv:
        count()
    tasks_table()
    models_table()
