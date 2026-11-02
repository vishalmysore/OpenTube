// Run one model stage over a JSONL file of candidates and append results to an output JSONL (resumable: ids already
// in the output are skipped). Every record logs wall-clock milliseconds and tokens, which is where the cost numbers
// on the "Receipts" page come from.
//
//   node pipeline/run_stage.mjs laya data/sample.jsonl data/out/laya_sample.jsonl
//   node pipeline/run_stage.mjs llm  data/sample.jsonl data/out/llm_sample.jsonl
//   node pipeline/run_stage.mjs cascade data/slice.jsonl data/out/cascade_slice.jsonl --lo 0.42 --hi 0.68
import fs from "node:fs";
import readline from "node:readline";
import { cleanComment, judgeState, RECOMMEND_Q } from "../web/text.js";
import { judgeBatch } from "../web/laya-batch.js";

const [mode, inPath, outPath, ...rest] = process.argv.slice(2);
const opt = (n, d) => { const i = rest.indexOf("--" + n); return i >= 0 ? rest[i + 1] : d; };
const BATCH = +opt("batch", 8), LO = +opt("lo", 0.4), HI = +opt("hi", 0.7), LIMIT = +opt("limit", Infinity);
const THREADS = +opt("threads", 2);

const done = new Set();
if (fs.existsSync(outPath)) for (const l of fs.readFileSync(outPath, "utf8").split("\n")) if (l) done.add(JSON.parse(l).id);
const items = [];
for await (const l of readline.createInterface({ input: fs.createReadStream(inPath) })) {
  if (!l) continue;
  const c = JSON.parse(l);
  if (done.has(c.id)) continue;
  c.clean ??= cleanComment(c.text);
  items.push(c);
  if (items.length >= LIMIT) break;
}
// Batch similar lengths together: a Laya batch is padded to its longest sequence, so mixing lengths wastes compute.
items.sort((a, b) => a.clean.length - b.clean.length);
console.log(`${mode}: ${items.length} to do, ${done.size} already done`);
const out = fs.createWriteStream(outPath, { flags: "a" });

let laya = null, llm = null;
if (mode === "laya" || mode === "cascade" || mode === "both") { const { loadLayaNode } = await import("./laya-node.mjs"); laya = await loadLayaNode(THREADS); }
if (mode === "llm" || mode === "cascade" || mode === "both") { const { loadLLM } = await import("./llm-node.mjs"); llm = await loadLLM(THREADS); }

const t0 = Date.now();
for (let i = 0; i < items.length; i += BATCH) {
  const batch = items.slice(i, i + BATCH);
  const recs = batch.map((c) => ({ id: c.id }));
  if (laya) {
    const { results, ms } = await judgeBatch(laya.ort, laya.session, laya.tok, laya.cfg, batch.map(judgeState), RECOMMEND_Q);
    results.forEach((r, k) => Object.assign(recs[k], { laya_p: r.probabilities.recommends, laya_tokens: r.tokens, laya_ms: Math.round(ms / batch.length) }));
  }
  for (let k = 0; k < batch.length; k++) {
    const r = recs[k];
    const needLLM = mode === "llm" || mode === "both" || (mode === "cascade" && r.laya_p > LO && r.laya_p < HI);
    if (mode === "cascade") r.route = r.laya_p >= HI ? "laya_accept" : r.laya_p <= LO ? "laya_reject" : "llm";
    if (needLLM) {
      const j = await llm.judge(judgeState(batch[k]));
      Object.assign(r, { llm_p: Math.round(j.p * 1e4) / 1e4, llm_tokens: j.promptTokens, llm_ms: Math.round(j.ms) });
    }
    out.write(JSON.stringify(r) + "\n");
  }
  const n = Math.min(i + BATCH, items.length), el = (Date.now() - t0) / 1000;
  if ((i / BATCH) % 10 === 0 || n === items.length) console.log(`${n}/${items.length}  ${el.toFixed(0)} s  (${(el / n).toFixed(2)} s/item)`);
}
out.end();
