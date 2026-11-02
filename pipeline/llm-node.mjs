// Stage 2 offline: Qwen2.5-1.5B-Instruct (ONNX q4 from onnx-community) run directly with ONNX Runtime on the CPU.
// One forward pass per comment; the label is p(yes) vs p(no) at the first answer token. Same prompt as the WebLLM
// page (web/llm-prompt.js), same weights family as WebLLM's Qwen2.5-1.5B-Instruct-q4f16_1-MLC.
import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
import { llmMessages, pYes } from "../web/llm-prompt.js";

const DIR = new URL("../models/local/onnx-community/Qwen2.5-1.5B-Instruct/", import.meta.url).pathname;

export function chatml(messages) {
  return messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}

export async function loadLLM(threads = 2) {
  const j = (f) => JSON.parse(fs.readFileSync(DIR + f, "utf8"));
  const cfg = j("config.json");
  const tok = new Tokenizer(j("tokenizer.json"), j("tokenizer_config.json"));
  // "basic" graph optimization: "all" more than doubles peak memory for this q4 graph.
  const session = await ort.InferenceSession.create(DIR + "onnx/model_q4.onnx", { graphOptimizationLevel: "basic", intraOpNumThreads: threads });
  const enc = (s) => Array.from(tok.encode(s, { add_special_tokens: false }).ids);
  const yes = [...new Set([enc("yes")[0], enc("Yes")[0]])], no = [...new Set([enc("no")[0], enc("No")[0]])];
  const kvH = cfg.num_key_value_heads, hd = cfg.hidden_size / cfg.num_attention_heads, layers = cfg.num_hidden_layers;
  const countTokens = (state) => enc(chatml(llmMessages(state))).length;

  async function judge(state) {
    const ids = enc(chatml(llmMessages(state)));
    const L = ids.length;
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, L]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from({ length: L }, (_, i) => BigInt(i)), [1, L]),
    };
    for (let i = 0; i < layers; i++) for (const kv of ["key", "value"]) feeds[`past_key_values.${i}.${kv}`] = new ort.Tensor("float32", new Float32Array(0), [1, kvH, 0, hd]);
    const t0 = performance.now();
    const out = await session.run(feeds, ["logits"]);
    const ms = performance.now() - t0;
    const V = out.logits.dims[2];
    const row = out.logits.data.subarray((L - 1) * V, L * V);
    let mx = -Infinity; for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
    let s = 0; for (let i = 0; i < V; i++) s += Math.exp(row[i] - mx);
    const lp = (set) => Math.log(set.reduce((a, i) => a + Math.exp(row[i] - mx), 0) / s);
    return { p: pYes(lp(yes), lp(no)), promptTokens: L, ms };
  }
  return { judge, countTokens };
}
