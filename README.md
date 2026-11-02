# OpenTube Graph

**An open, measured HackerTube.** Andrej Karpathy built [HackerTube](https://hackertube.org) in about two hours: the YouTube videos people recommend in Hacker News comments, filtered by an LLM. It's a lovely idea. This repository rebuilds it in the open and publishes the three things a two-hour build leaves out:

1. **How accurate the filter is**, measured against a labelled random sample, with a cross-validated estimate rather than only the in-sample one.
2. **What each stage costs**, as wall-clock time and tokens per stage, plus a projection for the whole archive.
3. **A knowledge graph** linking video, topic, HN thread and commenter, with **GraphRAG Q&A that runs in your browser** (MiniLM retrieval, SAGE-style graph expansion, WebLLM answers).

The filter is a cascade. **Laya**, a typed-decision model, makes the cheap yes/no call on "is this comment recommending the video?". **Qwen2.5-1.5B-Instruct** only sees the comments Laya is unsure about. That gives the cost comparison for free.

**Live demo:** https://vishalmysore.github.io/OpenTube/ · **Article:** [docs/article.md](docs/article.md)

| Page | What it does |
|---|---|
| `index.html` · Watch | Browse the re-linked canon by topic, with the HN comments that recommended each video and which model decided |
| `ask.html` · Ask the graph | Question → MiniLM embedding (in the browser) → seed videos → expansion along shared-thread, shared-commenter and similarity edges → WebLLM writes a cited answer |
| `receipts.html` · Receipts | The funnel, the accuracy table (with adjustable thresholds), cost per stage at your own prices, and a live Laya → WebLLM run on any comment |
| `label.html` · Labels | All 300 labelled comments with the rubric; change a label and the Receipts page re-scores with your version |

## Results

<!-- RESULTS:START -->
**Stage 0, the whole archive:** 49,324,111 HN items in 237 monthly files (12.4 GB), read in about 11 minutes on 2 vCPUs. That gives 257,038 comments linking 185,909 distinct YouTube videos; 1,345 videos were linked by 10 or more different people.

**Accuracy on 300 labelled random comments** (103 recommend a video):

| Filter | Accuracy, in-sample | Accuracy, cross-validated | Precision | Recall | AUROC | Time / comment | Sent to LLM |
|---|---|---|---|---|---|---|---|
| Laya only | 72.7% | 70.8% | 76.9% | 29.1% | 0.72 | 1.28 s | 0% |
| Qwen2.5-1.5B only | 71.0% | 67.7% | 59.1% | 50.5% | 0.72 | 4.11 s | 100% |
| **Cascade** | **74.7%** | **71.2%** | 67.5% | 50.5% | – | 2.46 s | 28.7% |

Answering "not a recommendation" every time scores 65.7%, so this is a hard, fuzzy call. The 1.5B LLM run on every comment is *worse* than Laya alone and about 3× slower. The cascade matches or beats both while calling the LLM on about a third of comments. Of the Laya question phrasings tried, the two-option question worked best (AUROC 0.72; three options 0.65, yes/no 0.68).

**This is a minimal run.** The canon cascade was stopped after 632 of 2,683 comments, covering 540 of the 1,345 canon videos (145 kept). Laya decided 479 on its own and sent 153 to the LLM. Comments were processed shortest first, so this slice leans toward bare links, and its keep rate is lower than the full canon's would be. `run_stage.mjs` resumes where it stopped.

**Cost, projected for all 257,038 comments on this CPU:** LLM on everything about 12 days; cascade about 6 days; Laya alone about 4 days. Exact token counts and editable $/token prices are on the Receipts page.
<!-- RESULTS:END -->

## The pipeline

```
open-index/hacker-news (237 monthly Parquet files, 12.4 GB)
  └─ stage 0  pipeline/extract.py            regex: comments with a YouTube link          free
       └─ pipeline/build_candidates.py       walk each comment up to its thread title
            ├─ data/sample.jsonl             300 random comments → labelled (data/labels.json)
            └─ data/slice_canon.jsonl        videos linked by ≥10 people, 2 comments each
                 └─ stage 1  Laya (ONNX, CPU)          p(recommends); confident? decide
                      └─ stage 2  Qwen2.5-1.5B (ONNX)  only when Laya is unsure
                           └─ pipeline/build_site_data.mjs  MiniLM vectors, topics, graph edges, receipts → web/data/
```

| Piece | File | Notes |
|---|---|---|
| Stage 0 | `pipeline/extract.py` | Streams one month at a time and deletes it after reading. Handles HN's escaped link targets (`http:&#x2F;&#x2F;…`). |
| Threads | `pipeline/build_candidates.py` | Vectorised parent-chain walk over all 41M comments |
| Stage 1 | `web/laya-core.js`, `web/laya-batch.js`, `pipeline/laya-node.mjs` | Same Laya port as layaAsRagJudge; batched across comments for the offline run |
| Stage 2 | `web/llm-prompt.js`, `pipeline/llm-node.mjs`, `web/webllm.js` | One prompt shared by the offline run (ONNX Runtime) and the page (WebLLM). The answer is p(yes) from the first token, not parsed text |
| Runner | `pipeline/run_stage.mjs` | `laya`, `llm`, `both` or `cascade`; resumable; logs ms and tokens per comment |
| Metrics | `web/metrics.js`, `pipeline/cv.py` | Accuracy with Wilson intervals, precision/recall/F1, AUROC; repeated 2-fold CV of the threshold choice |
| Graph | `pipeline/build_site_data.mjs`, `web/sage.js` | Similarity edges kept above the 98th percentile (max 6 per video), plus `thread` and `person` edges. Retrieval follows SAGE (seed selection + neighbour expansion) from [sagejava](https://github.com/vishalmysore/sagejava) |

### Reproduce

```bash
npm ci                                   # onnxruntime-node needs ONNXRUNTIME_NODE_INSTALL_CUDA=skip on machines without CUDA
pip install pyarrow numpy
bash scripts/download_models.sh          # Laya, Qwen2.5-1.5B (ONNX q4), MiniLM: ~2.3 GB
python3 -I pipeline/extract.py --out data/work                 # ~11 min, ~12 GB streamed, ~1 GB kept
python3 -I pipeline/build_candidates.py                        # data/candidates.jsonl
node pipeline/run_stage.mjs both data/sample.jsonl data/out/sample_both.jsonl
python3 -I pipeline/cv.py
node pipeline/run_stage.mjs cascade data/slice_canon.jsonl data/out/cascade_canon.jsonl --lo 0.53 --hi 0.61
node pipeline/build_site_data.mjs && node scripts/prepare_site.mjs
```

The sample and the canon slice are drawn with fixed seeds; the scripts that drew them are in the commit history and in `data/slice_canon_meta.json`.

**The whole archive on free CI.** `.github/workflows/full-archive.yml` splits the ~257K comments by month and runs the cascade as one GitHub Actions job per month (max 20 at a time), so no job comes near the 6-hour limit. It has not been run at full scale yet.

## Caveats

- **The labels are a first pass by Claude**, applying the rubric on the Labels page to 300 random comments, with 32 flagged as borderline. One labeller and 300 comments make a smoke test, not a benchmark. Corrections are welcome: export from the Labels page and open a PR.
- **Thresholds are fitted on the same sample.** That's why the cross-validated numbers are published next to the in-sample ones.
- **Only the canon was run.** The full-archive cost rows multiply measured per-comment rates by exact token counts.
- **A video is kept if either of its two sampled comments recommends it.** "Re-linked by 10+ people" is a strong prior on its own, so the site shows the filtered-out videos too (Watch → "Filtered out").
- **Titles come from noembed.com in the viewer's browser.** The HN data has only video IDs; nothing from YouTube is stored here.
- **WebLLM needs WebGPU** (Chrome or Edge on desktop). Without it, Ask still shows the retrieved videos and the graph, and Receipts can still run Laya.

## Credits

Inspired by Andrej Karpathy's HackerTube (not affiliated). Data: [open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) (ODC-By). Models: [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) via [layaForWebTrained](https://huggingface.co/VishalMysore/layaForWebTrained), Qwen2.5-1.5B-Instruct via [WebLLM](https://github.com/mlc-ai/web-llm) and [onnx-community](https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct), [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2). See `NOTICE.md`. Code: Apache-2.0.
