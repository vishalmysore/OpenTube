// all-MiniLM-L6-v2 in Node, numerically the same recipe as web/embedder.js (mean pooling over the attention mask,
// then L2 normalization), so vectors built offline match queries embedded in the browser.
import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";

const DIR = new URL("../models/local/Xenova/all-MiniLM-L6-v2/", import.meta.url).pathname;
const D = 384, MAXTOK = 256;

export async function loadEmbedderNode(threads = 2) {
  const j = (f) => JSON.parse(fs.readFileSync(DIR + f, "utf8"));
  const tok = new Tokenizer(j("tokenizer.json"), j("tokenizer_config.json"));
  const session = await ort.InferenceSession.create(DIR + "onnx/model_quantized.onnx", { intraOpNumThreads: threads });
  const needsTypes = session.inputNames.includes("token_type_ids");
  const encode = (t) => { let ids = tok.encode(String(t)).ids; if (ids.length > MAXTOK) ids = [...ids.slice(0, MAXTOK - 1), ids.at(-1)]; return Array.from(ids); };
  async function run(texts) {
    const enc = texts.map(encode);
    const n = enc.length, L = Math.max(...enc.map((e) => e.length));
    const ids = new BigInt64Array(n * L), mask = new BigInt64Array(n * L);
    enc.forEach((e, i) => e.forEach((t, k) => { ids[i * L + k] = BigInt(t); mask[i * L + k] = 1n; }));
    const feeds = { input_ids: new ort.Tensor("int64", ids, [n, L]), attention_mask: new ort.Tensor("int64", mask, [n, L]) };
    if (needsTypes) feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(n * L), [n, L]);
    const out = await session.run(feeds);
    const h = (out.last_hidden_state || out[session.outputNames[0]]).data;
    return enc.map((e, i) => {
      const v = new Float32Array(D);
      for (let k = 0; k < e.length; k++) for (let d = 0; d < D; d++) v[d] += h[(i * L + k) * D + d];
      let norm = 0; for (let d = 0; d < D; d++) { v[d] /= e.length; norm += v[d] * v[d]; }
      norm = Math.sqrt(norm) || 1; for (let d = 0; d < D; d++) v[d] /= norm;
      return v;
    });
  }
  async function embed(texts, batch = 16) {
    const out = [];
    for (let i = 0; i < texts.length; i += batch) out.push(...await run(texts.slice(i, i + batch)));
    return out;
  }
  return { embed, dim: D };
}
