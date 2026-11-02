#!/usr/bin/env bash
# Fetch the three models the offline pipeline uses into models/ (about 2.3 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
L=models/laya; mkdir -p "$L"
B=https://huggingface.co/VishalMysore/layaForWebTrained/resolve/main
for f in manifest.json tokenizer.json tokenizer_config.json rl_agent_config.json laya_q8e8.onnx; do curl -sfL -o "$L/$f" "$B/$f"; done
rm -f "$L/laya_q8e8.onnx.data"
for i in $(seq -w 0 17); do curl -sfL "$B/laya_q8e8.onnx.data.part0$i" >> "$L/laya_q8e8.onnx.data"; done
Q=models/local/onnx-community/Qwen2.5-1.5B-Instruct; mkdir -p "$Q/onnx"
B=https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main
for f in config.json tokenizer.json tokenizer_config.json onnx/model_q4.onnx; do curl -sfL -o "$Q/$f" "$B/$f"; done
E=models/local/Xenova/all-MiniLM-L6-v2; mkdir -p "$E/onnx"
B=https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9
for f in tokenizer.json tokenizer_config.json onnx/model_quantized.onnx; do curl -sfL -o "$E/$f" "$B/$f"; done
echo "models ready"
