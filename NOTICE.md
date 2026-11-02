# Notice

OpenTube Graph is an independent project inspired by Andrej Karpathy's HackerTube. It is not affiliated with or endorsed by Andrej Karpathy, Y Combinator / Hacker News, YouTube / Google, ConvAI Innovations, TypeSafe, Alibaba (Qwen), the MLC / WebLLM authors, Hugging Face, Microsoft, or the sentence-transformers authors.

## This project

The pipeline, pages, labels and generated site data in this repository are Copyright 2026 vishalmysore and licensed under the Apache License, Version 2.0 (see `LICENSE`).

## Data

- **Hacker News archive:** [`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news) on Hugging Face, released under the Open Data Commons Attribution License (ODC-By). `web/data/` contains short excerpts of Hacker News comments (attributed by username and linked to the original item) and thread titles derived from it. Hacker News content belongs to its authors.
- **YouTube:** only video IDs are stored. Thumbnails load from `i.ytimg.com` and titles from `noembed.com` in the viewer's browser at view time; nothing from YouTube is redistributed.

## Models (downloaded at runtime or by `scripts/download_models.sh`; not part of this repository)

- **Stage 1 judge:** [`VishalMysore/layaForWebTrained`](https://huggingface.co/VishalMysore/layaForWebTrained), a modified derivative of [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) (Copyright ConvAI Innovations, Apache-2.0), exported to ONNX by the [layaForWeb](https://github.com/vishalmysore/layaForWeb) project. Laya is built on ModernBERT-large by Answer.AI and LightOn (Apache-2.0).
- **Stage 2 judge and answer writer:** Qwen2.5-1.5B-Instruct (Qwen team, Apache-2.0): `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` through [WebLLM](https://github.com/mlc-ai/web-llm) (Apache-2.0) in the browser, and [`onnx-community/Qwen2.5-1.5B-Instruct`](https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct) offline.
- **Retriever:** [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (Apache-2.0), the ONNX export of `sentence-transformers/all-MiniLM-L6-v2`.

## Code reused from the author's other projects

- `web/laya-core.js`, `web/model.js`, `web/embedder.js`, `web/coi.js` and `web/coi-sw.js` come from [layaAsRagJudge](https://github.com/vishalmysore/layaAsRagJudge) (originally layaForWeb / layaForWorkflows). `laya-core.js` is a JavaScript port of `laya/common.py` and `laya/agent.py` from https://github.com/NandhaKishorM/laya (Apache-2.0). `coi-sw.js` was changed to send `Cross-Origin-Embedder-Policy: credentialless`.
- `web/sage.js` follows the SAGE retrieval algorithm (Titiya et al., arXiv:2602.16964) as implemented in [sagejava](https://github.com/vishalmysore/sagejava).
- `web/webllm.js` loads WebLLM the same way as [recursiveMASWebLLM](https://github.com/vishalmysore/recursiveMASWebLLM).

## Third-party software shipped with the site (`vendor/`)

- **ONNX Runtime Web** (`onnxruntime-web` 1.30.0): Copyright (c) Microsoft Corporation, MIT License. `licenses/onnxruntime-LICENSE.txt`; notices for components inside the WebAssembly binary are in `licenses/onnxruntime-ThirdPartyNotices.txt`.
- **Tokenizers.js** (`@huggingface/tokenizers` 0.2.0): Hugging Face, Apache License 2.0. `licenses/tokenizers.js-LICENSE.txt`.
- **WebLLM** is loaded at runtime from `esm.run` (jsDelivr), not vendored.
