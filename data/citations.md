# Citation sources

Every task and model citation, with the record it was checked against (2026-11-11).


All new entries are in `task_refs.bib`. Keys that already exist in the paper's `references.bib` are reused: `thakur2021beir`, `su2024bright`, `enevoldsen2025mmteb`. None of the 19 new keys clash with the existing bib.

Verification sources: ACL Anthology `.bib` exports; Crossref metadata via doi.org content negotiation (ACM, Springer, Elsevier, KDD); Springer chapter and book pages (for LNCS volume and pages); the CEUR-WS volume index; arXiv abstract pages; the NeurIPS Datasets & Benchmarks proceedings page and its `.bib`; and the Zenodo record API. DBLP rate-limited every request (HTTP 429), so none of the entries below rely on it.

| mteb task | bib key(s) | verified against | discrepancy with mteb's BibTeX |
|---|---|---|---|
| ArguAna | `wachsmuth2018arguana` (+ `thakur2021beir`) | https://aclanthology.org/P18-1023.bib | mteb's booktitle is just "ACL", and it has no pages or DOI. The paper itself is correct. |
| ClimateFEVERHardNegatives | `diggelmann2020climatefever` + `enevoldsen2025mmteb` (HN version) | https://arxiv.org/abs/2012.00614 | mteb gives the year as 2021. That is the date of v2 (2 Jan 2021); v1 is from 1 Dec 2020. The paper was never formally published; arXiv lists it as accepted at the NeurIPS 2020 "Tackling Climate Change with ML" workshop, which is non-archival. The authors are correct. |
| CQADupstackGamingRetrieval | `hoogeveen2015cqadupstack` (+ `thakur2021beir`) | https://doi.org/10.1145/2838931.2838934 (Crossref) | The paper is correct. mteb's pages are `3:1--3:8` (article 3) and Crossref's are 1–8; I used articleno=3 with pages 1--8. The key `hoogeveen2015` has no descriptive suffix. |
| CQADupstackUnixRetrieval | `hoogeveen2015cqadupstack` | same | same |
| FEVERHardNegatives | `thorne2018fever` + `enevoldsen2025mmteb` | https://aclanthology.org/N18-1074.bib | None (it matches the Anthology). |
| FiQA2018 | `maia2018fiqa` (+ `thakur2021beir`) | https://doi.org/10.1145/3184558.3192301 (Crossref) | **Wrong paper.** mteb cites BEIR (`thakur2021beir`) rather than the paper that introduced the dataset: Maia et al., WWW'18 Companion, pp. 1941–1942. |
| HotpotQAHardNegatives | `yang2018hotpotqa` + `enevoldsen2025mmteb` | https://aclanthology.org/D18-1259.bib | None. |
| SCIDOCS | `cohan2020specter` (+ `thakur2021beir`) | https://aclanthology.org/2020.acl-main.207.bib | The paper is correct. mteb's booktitle is just "ACL", with no pages or DOI. |
| Touche2020Retrieval.v3 | `bondarenko2020touche` + `thakur2024touche` | https://doi.org/10.1007/978-3-030-58219-7_26 and the Springer chapter page (LNCS 12260, pp. 384–395); https://doi.org/10.1145/3626772.3657861 (Crossref) | mteb cites only Thakur et al. 2024, which does describe v3. It does not cite the Touché 2020 overview that introduced the dataset. Its SIGIR entry also has no pages (1420–1430) or DOI, and uses a non-standard `address_` field. A CEUR working-notes version of the overview also exists (Vol-2696); I chose the LNCS version. |
| TRECCOVID | `voorhees2020treccovid` + `roberts2021treccovid` (+ `thakur2021beir`) | https://doi.org/10.1145/3451964.3451965 and https://doi.org/10.1016/j.jbi.2021.103865 (Crossref) | mteb's Roberts 2021 entry is correct, but it leaves out the collection paper (Voorhees et al., SIGIR Forum 54(1), 2020), so I added it. |
| BrightAopsRetrieval | `su2024bright` (existing) | not re-verified (key already in the bib) | — |
| BrightBiologyRetrieval | `su2024bright` | — | — |
| BrightEarthScienceRetrieval | `su2024bright` | — | — |
| BrightEconomicsRetrieval | `su2024bright` | — | — |
| BrightLeetcodeRetrieval | `su2024bright` | — | — |
| BrightPonyRetrieval | `su2024bright` | — | — |
| BrightPsychologyRetrieval | `su2024bright` | — | — |
| BrightRoboticsRetrieval | `su2024bright` | — | — |
| BrightStackoverflowRetrieval | `su2024bright` | — | — |
| BrightSustainableLivingRetrieval | `su2024bright` | — | — |
| BrightTheoremQAQuestionsRetrieval | `su2024bright` | — | — |
| BrightTheoremQATheoremsRetrieval | `su2024bright` | — | — |
| AILACasedocs | `bhattacharya2019aila` + `bhattacharya2020ailadata` | https://ceur-ws.org/Vol-2517/ (the table of contents lists it as pp. 1–12; T1-1.pdf resolves) and https://zenodo.org/api/records/4063986 (published 2020-10-02) | mteb's Zenodo dataset entry is accurate (authors, Oct 2020, DOI). It does not cite the peer-reviewed track overview, CEUR-WS Vol-2517, so I added it. |
| AppsRetrieval | `hendrycks2021apps` + `li2025coir` | APPS page and .bib at https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html ; CoIR at https://aclanthology.org/2025.acl-long.1072.bib | mteb's booktitle ("Thirty-fifth Conference ... (Round 2)") is not the official one, which is "Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks", vol. 1. The mteb task loads `CoIR-Retrieval/apps` (checked in mteb's `apps_retrieval.py`), but its citation does not mention CoIR. |
| CUREv1 | `sheikh2025cure` | https://arxiv.org/abs/2412.06954 (lists Related DOI 10.1145/3711896.3737435) and https://doi.org/10.1145/3711896.3737435 (Crossref: KDD '25 V.2, pp. 5270–5277); the HF dataset card links the same arXiv paper | **mteb has no citation.** The two sources split the first author's name differently: arXiv has "Sheikh, Nadia Athar" and Crossref has "Athar Sheikh, Nadia". I used the arXiv form; check the paper's own byline if that matters. |
| FinQARetrieval | `chen2021finqa` | https://aclanthology.org/2021.emnlp-main.300.bib | mteb uses `@article` with `journal = {Proceedings of ... EMNLP}`, which is the wrong entry type, and has no pages, publisher or DOI. The paper is correct. Note that mteb's data comes from `embedding-benchmark/FinQA`, a retrieval conversion; I found no separate paper for it. |
| NFCorpus | `boteva2016nfcorpus` (+ `thakur2021beir`) | https://doi.org/10.1007/978-3-319-30671-1_58 and the Springer chapter page (ECIR 2016, LNCS 9626, pp. 716–722) | The paper is correct, but the entry is malformed: an `@inproceedings` with a `journal` field and no booktitle, plus non-standard `city`, `country` and `journal-abbrev` fields, and no pages or DOI. |
| SciFact | `wadden2020scifact` (+ `thakur2021beir`) | https://aclanthology.org/2020.emnlp-main.609.bib | **Wrong paper.** mteb cites SPECTER (`specter2020cohan`) instead of Wadden et al., EMNLP 2020, "Fact or Fiction: Verifying Scientific Claims", pp. 7534–7550. |

## Unverified
- BRIGHT (`su2024bright`): I did not re-check it, because the brief says the key is already in the paper's bib. The paper's existing entry and mteb's agree on the essentials: ICLR 2025, arXiv:2407.12883.
- Everything else was verified against a primary record. DBLP was unusable (rate-limited), so none of the entries were cross-checked against it.


Bib file: `model_refs.bib` (23 entries, 28 models). No key clashes with the existing paper bib.
Method: the model card's README was downloaded from huggingface.co. Then the canonical record was checked: the arXiv abs page plus arxiv.org/bibtex, the ACL Anthology `.bib`, Crossref for the ACM DOI, the NeurIPS proceedings `.bib`, and the TMLR listing with its `.bib` at jmlr.org/tmlr. DBLP, Semantic Scholar and the OpenReview API were blocked (challenge pages or rate limits), so they were not used.

| HF id | bib key(s) | source verified | paper? | discrepancies vs mteb citation |
|---|---|---|---|---|
| Alibaba-NLP/gte-Qwen2-7B-instruct | li2023gte | arxiv.org/abs/2308.03281; card Citation section | paper (arXiv only; no venue found) | None. Note: the paper predates the Qwen2 model and does not describe it, but it is what the card asks users to cite. |
| BAAI/bge-base-en-v1.5 | xiao2024cpack | doi 10.1145/3626772.3657878 (Crossref); arXiv 2309.07597 ("SIGIR 2024") | paper (SIGIR 2024) | mteb matches. The card cites an older arXiv version with a different title ("C-Pack: Packaged Resources To Advance General Chinese Embedding") and only 4 authors. Use the SIGIR version. |
| BAAI/bge-large-en-v1.5 | xiao2024cpack | as above | paper | as above |
| BAAI/bge-small-en-v1.5 | xiao2024cpack | as above | paper | as above |
| IEITYuan/Yuan-embedding-2.0-en | ieityuan2025yuanembedding | huggingface.co/IEITYuan/Yuan-embedding-2.0-en (created 2025-11-27) | **no paper** (card has no citation; arXiv search finds nothing) | mteb has none. Consistent. |
| Linq-AI-Research/Linq-Embed-Mistral | choi2024linq | arxiv.org/abs/2412.03223 ("Linq-Embed-Mistral Technical Report", submitted 4 Dec 2024) | paper (arXiv tech report) | mteb has none, but an arXiv report exists. The card cites a 2024 blog post with a different title and author order (Kim first); arXiv lists Choi first. |
| Qwen/Qwen3-Embedding-0.6B | zhang2025qwen3embedding | arxiv.org/abs/2506.05176; card | paper (arXiv) | None |
| Qwen/Qwen3-Embedding-4B | zhang2025qwen3embedding | as above | paper | None |
| Qwen/Qwen3-Embedding-8B | zhang2025qwen3embedding | as above | paper | None |
| Snowflake/snowflake-arctic-embed-l-v2.0 | yu2024arcticembed2 | arxiv.org/abs/2412.04506; card links it as the tech report | paper (arXiv; no venue found) | None |
| codefuse-ai/F2LLM-v2-4B | zhang2026f2llmv2 | arxiv.org/abs/2603.19223; card | paper (arXiv) | None |
| colbert-ir/colbertv2.0 | santhanam2022colbertv2 | aclanthology.org/2022.naacl-main.272.bib; arXiv 2112.01488 ("NAACL 2022") | paper (NAACL 2022) | **mteb has none** |
| google/embeddinggemma-300m | vera2025embeddinggemma | arxiv.org/abs/2509.20354 + arxiv bibtex (89 authors); HF API tag arxiv:2509.20354 (the card itself is gated) | paper (arXiv) | None |
| ibm-granite/granite-embedding-english-r2 | awasthy2025granite | arxiv.org/abs/2508.21085; card "Paper:" link | paper (arXiv) | None (the card uses @misc, mteb uses @article; same content) |
| intfloat/e5-mistral-7b-instruct | wang2024e5mistral | aclanthology.org/2024.acl-long.642.bib; arXiv 2401.00368 ("Accepted by ACL 2024") | paper (ACL 2024) | mteb's ACL entry is correct. It also adds the original E5 paper (wang2022text, arXiv 2212.03533), which is the base E5 work and not the paper that introduced this model. I left it out. The card itself still cites the arXiv preprint. |
| intfloat/multilingual-e5-large-instruct | wang2024multilinguale5 | arxiv.org/abs/2402.05672; card | paper (arXiv tech report) | None |
| jinaai/jina-embeddings-v5-text-nano | akram2026jinav5 | arxiv.org/abs/2602.15547; card | paper (arXiv) | None |
| jinaai/jina-embeddings-v5-text-small | akram2026jinav5 | as above | paper | None |
| lightonai/DenseOn | sourty2026denseon | arxiv.org/abs/2607.27178 (submitted 29 Jul 2026) | paper (arXiv) | mteb matches arXiv. **The card is stale**: it still cites the HF blog with a different title ("...Open State-of-the-Art Single and Multi-Vector Models") and a different author list (includes Orion Weller; "Paulo Demoura"). The arXiv paper has 4 authors, without Weller. |
| microsoft/harrier-oss-v1-0.6b | microsoft2026harrier | huggingface.co/microsoft/harrier-oss-v1-0.6b; sibling cards (270m, 27b); Bing blog (Apr 2026) | **no paper** (no citation on the card; arXiv search for "harrier" finds nothing; the blog only cites E5 papers) | mteb has none. Consistent. |
| minishlab/potion-retrieval-32M | tulkens2024model2vec | card citation section; Zenodo record (concept DOI 10.5281/zenodo.17270888, Tulkens & van Dongen) | **no paper**: software citation the card asks for (Model2Vec) | **mteb has none.** Converted the card's `@software` to `@misc` because acl_natbib.bst has no @software type. |
| mixedbread-ai/mxbai-embed-large-v1 | lee2024mxbai, li2024angle | mixedbread.com/blog/mxbai-embed-large-v1 (published 2024-03-08); aclanthology.org/2024.acl-long.101.bib; arXiv 2309.12871 ("Published at ACL24") | blog (no model paper) + AnglE paper (ACL 2024) | **mteb gives only the AnglE arXiv preprint (2023)**. The peer-reviewed version is ACL 2024 under the title "AoE: Angle-optimized Embeddings for Semantic Textual Similarity". mteb also leaves out the blog citation the card asks for. The card's blog title says "Embeddings Model", but the blog page says "Embedding Model", and I used the blog's wording. |
| mteb/baseline-bm25s | lu2024bm25s | arxiv.org/abs/2407.03618 ("Technical Report") | paper (arXiv) | None |
| nomic-ai/nomic-embed-text-v1.5 | nussbaum2025nomic | jmlr.org/tmlr/papers listing + bib/IPmzyQSiQE.bib (Feb 2025, Reproducibility Certification) | paper (TMLR 2025) | mteb matches TMLR. The card still cites the 2024 arXiv preprint, and its author order (Morris, Duderstadt, Mulyar) differs from TMLR (Morris, Mulyar, Duderstadt). I used the TMLR order. |
| nomic-ai/nomic-embed-text-v2-moe | nussbaum2025moe | arxiv.org/abs/2502.07972; card | paper (arXiv) | None |
| sentence-transformers/all-MiniLM-L6-v2 | reimers2019sbert, wang2020minilm | aclanthology.org/D19-1410.bib; NeurIPS 2020 proceedings .bib (3f5ee243...) | **no model paper**: the card has no citation, so I cite the framework (SBERT) and the base model (MiniLM) | mteb's SBERT entry has a truncated booktitle (it leaves out "and the 9th IJCNLP (EMNLP-IJCNLP)"), its url points to arXiv, it has no pages or DOI, and it gives month "11". mteb does not cite MiniLM. |
| thenlper/gte-large | li2023gte | arxiv.org/abs/2308.03281; card | paper (arXiv only) | None |
| thenlper/gte-small | li2023gte | as above | paper | None |

## Unverifiable / caveats
- **No peer-reviewed venue could be confirmed** for GTE, Arctic-Embed 2.0, Qwen3 Embedding, EmbeddingGemma, Granite R2, mE5, jina v5, F2LLM-v2, DenseOn, BM25S or nomic v2-moe. The arXiv pages show no journal-ref, and web search found no proceedings. DBLP was not reachable (bot challenge), so a recent acceptance could have been missed. These are cited as arXiv preprints.
- The **Yuan-embedding-2.0-en** and **harrier-oss-v1** cards have no paper. Both get a model-card `@misc`, with years 2025 and 2026 taken from the HF creation dates.
- The **Model2Vec** year (2024) follows the card's requested citation. The Zenodo concept DOI covers all versions; the latest version (v0.10.0) is dated 2026-10-11.
