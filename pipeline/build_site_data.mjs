// Turn the pipeline outputs into the static files the site reads (web/data/*). Nothing here calls a model except the
// MiniLM embedder (video vectors + topic assignment) and the two tokenizers (token counts for the cost projection).
//
//   node pipeline/build_site_data.mjs
import fs from "node:fs";
import readline from "node:readline";
import { Tokenizer } from "@huggingface/tokenizers";
import { cleanComment, window, judgeState } from "../web/text.js";
import { llmMessages } from "../web/llm-prompt.js";
import { chatml } from "./llm-node.mjs";
import { TOPICS } from "../web/topics.js";
import { loadEmbedderNode } from "./embed-node.mjs";
import { evaluate, cascadeDecision } from "../web/metrics.js";

const R = (p) => new URL("../" + p, import.meta.url).pathname;
const readJsonl = (p) => fs.existsSync(R(p)) ? fs.readFileSync(R(p), "utf8").split("\n").filter(Boolean).map(JSON.parse) : [];
const OUT = R("web/data/");
fs.mkdirSync(OUT, { recursive: true });
const TH = JSON.parse(fs.readFileSync(R("data/thresholds.json"), "utf8"));

// ---------- stage 0 ----------
const xlog = JSON.parse(fs.readFileSync(R("data/work/extract_log.json"), "utf8"));
const months = Object.values(xlog);
const sum = (k) => months.reduce((a, m) => a + m[k], 0);

// ---------- all candidates: per-video aggregates + token counts ----------
const j = (p) => JSON.parse(fs.readFileSync(R(p), "utf8"));
const qwenTok = new Tokenizer(j("models/local/onnx-community/Qwen2.5-1.5B-Instruct/tokenizer.json"), j("models/local/onnx-community/Qwen2.5-1.5B-Instruct/tokenizer_config.json"));
const layaTokenizer = new Tokenizer(j("models/laya/tokenizer.json"), j("models/laya/tokenizer_config.json"));
const meta = j("data/slice_canon_meta.json");
const canonSet = new Set(Object.keys(meta.per_video));
const byVideo = new Map(); // canon video -> {people:Set, comments:[], threads:Map}
for (const v of canonSet) byVideo.set(v, { people: new Set(), comments: [], threads: new Map() });

let nCand = 0, qwenTokens = 0, layaTokens = 0;
const yearCounts = {};
const tokCache = R("data/token_counts.json");
const haveTok = fs.existsSync(tokCache);
for await (const l of readline.createInterface({ input: fs.createReadStream(R("data/candidates.jsonl")) })) {
  if (!l) continue;
  const c = JSON.parse(l);
  nCand++;
  yearCounts[c.time.slice(0, 4)] = (yearCounts[c.time.slice(0, 4)] || 0) + 1;
  const touches = c.video_ids.filter((v) => canonSet.has(v));
  if (!haveTok || touches.length) {
    c.clean = cleanComment(c.text);
    if (!haveTok) {
      const st = judgeState(c);
      qwenTokens += qwenTok.encode(chatml(llmMessages(st)), { add_special_tokens: false }).ids.length;
      // Laya sequence = question head + options (~70 tokens) + the serialized state.
      layaTokens += layaTokenizer.encode(JSON.stringify(st), { add_special_tokens: false }).ids.length + 72;
    }
  }
  for (const v of touches) {
    const b = byVideo.get(v);
    b.people.add(c.by);
    b.comments.push({ id: c.id, by: c.by, date: c.time, sid: c.story_id, text: window(c.clean, 360) });
    if (c.story_id) { const t = b.threads.get(c.story_id) || { sid: c.story_id, title: c.story_title, score: c.story_score, n: 0 }; t.n++; b.threads.set(c.story_id, t); }
  }
}
if (!haveTok) fs.writeFileSync(tokCache, JSON.stringify({ candidates: nCand, qwenTokens, layaTokens }));
const TOK = j("data/token_counts.json");
console.log("candidates", nCand, TOK);

// ---------- labelled sample ----------
const sampleRows = readJsonl("data/sample.jsonl");
const sampleRes = new Map(readJsonl("data/out/sample_both.jsonl").map((r) => [r.id, r]));
const labels = new Map(j("data/labels.json").map((x) => [x.id, x]));
const sample = sampleRows.filter((c) => sampleRes.has(c.id)).map((c) => {
  const r = sampleRes.get(c.id), lab = labels.get(c.id);
  return {
    id: c.id, by: c.by, date: c.time, thread: c.story_title, sid: c.story_id, video: c.video_ids[0],
    comment: window(cleanComment(c.text)), gold: lab.label, borderline: lab.borderline, labeler: lab.labeler,
    laya_p: r.laya_p, llm_p: r.llm_p, laya_ms: r.laya_ms, llm_ms: r.llm_ms, laya_tokens: r.laya_tokens, llm_tokens: r.llm_tokens,
  };
});
const accuracy = evaluate(sample, TH);
fs.writeFileSync(OUT + "sample.json", JSON.stringify(sample));

// ---------- canon cascade ----------
const casc = new Map(readJsonl("data/out/cascade_canon.jsonl").map((r) => [r.id, r]));
const routes = { laya_accept: 0, laya_reject: 0, llm: 0, llm_accept: 0 };
let layaMs = 0, llmMs = 0, layaTok = 0, llmTok = 0;
for (const r of casc.values()) {
  routes[r.route]++; layaMs += r.laya_ms; layaTok += r.laya_tokens;
  if (r.route === "llm") { llmMs += r.llm_ms; llmTok += r.llm_tokens; if (r.llm_p >= TH.llm) routes.llm_accept++; }
}

// ---------- videos ----------
const vids = [...canonSet].filter((v) => meta.per_video[v].some((id) => casc.has(id)));
const videos = vids.map((v) => {
  const b = byVideo.get(v);
  const sampled = meta.per_video[v].filter((id) => casc.has(id)).map((id) => {
    const r = casc.get(id), c = b.comments.find((x) => x.id === id);
    const d = cascadeDecision(r, TH);
    return { ...c, route: r.route, laya_p: r.laya_p, llm_p: r.llm_p ?? null, rec: d.recommends };
  });
  const others = b.comments.filter((c) => !meta.per_video[v].includes(c.id)).sort((x, y) => y.text.length - x.text.length).slice(0, 3);
  const threads = [...b.threads.values()].sort((x, y) => y.n - x.n || y.score - x.score);
  const dates = b.comments.map((c) => c.date).sort();
  return {
    v, people: b.people.size, links: b.comments.length, first: dates[0], last: dates.at(-1),
    rec: sampled.filter((s) => s.rec).length, sampled, others,
    threads: threads.slice(0, 8).map((t) => [t.sid, t.title, t.score, t.n]),
    _people: [...b.people], _threads: threads.map((t) => t.sid),
    _doc: [...threads.slice(0, 5).map((t) => t.title), ...b.comments.slice(0, 8).map((c) => c.text.replace(/\[YouTube video\]/g, ""))].join(" \n ").slice(0, 2400),
  };
});
console.log("videos", videos.length, "recommended", videos.filter((x) => x.rec > 0).length);

// ---------- embeddings, topics ----------
const E = await loadEmbedderNode(2);
const vecs = await E.embed(videos.map((x) => x._doc));
const tvecs = await E.embed(TOPICS.map((t) => t[2]));
const dotp = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
videos.forEach((x, i) => {
  const s = tvecs.map((t) => dotp(vecs[i], t));
  const order = s.map((_, k) => k).sort((a, b) => s[b] - s[a]);
  x.topic = TOPICS[order[0]][0];
  x.topic2 = s[order[1]] > s[order[0]] - 0.02 ? TOPICS[order[1]][0] : null;
});
const D = 384, q8 = new Int8Array(videos.length * D);
vecs.forEach((v, i) => v.forEach((x, d) => { q8[i * D + d] = Math.max(-127, Math.min(127, Math.round(x * 127))); }));
fs.writeFileSync(OUT + "vectors.i8", Buffer.from(q8.buffer));

// ---------- graph edges (SAGE: percentile-pruned similarity + structural edges) ----------
const N = videos.length, edges = [];
const sims = [];
for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) sims.push(dotp(vecs[a], vecs[b]));
const sorted = Float32Array.from(sims).sort();
const P = sorted[Math.floor(sorted.length * 0.98)]; // keep the top 2% of pairs, at most 6 per video
const top = Array.from({ length: N }, () => []);
for (let a = 0, k = 0; a < N; a++) for (let b = a + 1; b < N; b++, k++) if (sims[k] >= P) { top[a].push([b, sims[k]]); top[b].push([a, sims[k]]); }
const seen = new Set();
top.forEach((lst, a) => lst.sort((x, y) => y[1] - x[1]).slice(0, 6).forEach(([b, s]) => {
  const key = a < b ? `${a}-${b}` : `${b}-${a}`; if (seen.has(key)) return; seen.add(key);
  edges.push([Math.min(a, b), Math.max(a, b), "sim", +s.toFixed(3), null]);
}));
const addGroup = (kind, groups, maxSize, label) => {
  for (const [key, members] of groups) {
    if (members.length < 2 || members.length > maxSize) continue;
    const w = +(1 / Math.log2(members.length + 1)).toFixed(3);
    for (let x = 0; x < members.length; x++) for (let y = x + 1; y < members.length; y++) {
      const a = Math.min(members[x], members[y]), b = Math.max(members[x], members[y]);
      const k2 = `${kind}:${a}-${b}`; if (seen.has(k2)) continue; seen.add(k2);
      edges.push([a, b, kind, w, label(key)]);
    }
  }
};
const threadTitle = new Map(); videos.forEach((x) => x.threads.forEach(([sid, t]) => threadTitle.set(sid, t)));
const tg = new Map(), pg = new Map();
videos.forEach((x, i) => { x._threads.forEach((s) => tg.set(s, [...(tg.get(s) || []), i])); x._people.forEach((p) => pg.set(p, [...(pg.get(p) || []), i])); });
addGroup("thread", tg, 12, (sid) => [sid, threadTitle.get(sid) || ""]);
addGroup("person", pg, 8, (p) => p);
console.log("edges", edges.length, { sim: edges.filter((e) => e[2] === "sim").length, thread: edges.filter((e) => e[2] === "thread").length, person: edges.filter((e) => e[2] === "person").length });

for (const x of videos) { delete x._doc; delete x._people; delete x._threads; }
fs.writeFileSync(OUT + "videos.json", JSON.stringify(videos));
fs.writeFileSync(OUT + "graph.json", JSON.stringify({ n: N, dim: D, percentile: 0.98, edges }));

// ---------- receipts ----------
const sampleMean = (k) => sample.reduce((a, s) => a + s[k], 0) / sample.length;
const receipts = {
  built: new Date().toISOString().slice(0, 10),
  hardware: "2 vCPU cloud container, no GPU (ONNX Runtime, CPU)",
  thresholds: TH,
  stage0: {
    months: months.length, items: sum("items"), comments: sum("comments"), yt_comments: sum("yt_comments"),
    bytes: sum("bytes"), download_s: +sum("download_s").toFixed(0), process_s: +sum("process_s").toFixed(0),
    distinct_videos: j("data/stage0_videos.json").distinct, canon_videos: canonSet.size, by_year: yearCounts,
  },
  canon: {
    rule: meta.rule, canon_total: canonSet.size, judged_note: "minimal run: stopped after the first comments; videos with at least one judged comment are shown", comments: casc.size, videos: videos.length, recommended_videos: videos.filter((x) => x.rec > 0).length,
    routes, laya_s: +(layaMs / 1000).toFixed(0), llm_s: +(llmMs / 1000).toFixed(0), laya_tokens: layaTok, llm_tokens: llmTok,
  },
  per_item: {
    laya_ms: +sampleMean("laya_ms").toFixed(0), llm_ms: +sampleMean("llm_ms").toFixed(0),
    laya_tokens: +sampleMean("laya_tokens").toFixed(0), llm_tokens: +sampleMean("llm_tokens").toFixed(0),
  },
  full_archive: { candidates: nCand, qwen_prompt_tokens: TOK.qwenTokens, laya_tokens: TOK.layaTokens },
  accuracy,
  cv: j("data/cv.json"),
  phrasings: { choice2: accuracy.summary.laya.auroc, choice3: 0.648, noul: 0.683, choice2short: 0.710 },
};
fs.writeFileSync(OUT + "receipts.json", JSON.stringify(receipts, null, 1));
console.log(JSON.stringify(receipts.accuracy.summary, null, 1));
