// Try several phrasings of the stage-1 question on the labelled sample and report AUROC for each.
// Laya's probabilities are sensitive to wording (layaAsRagJudge found the same), so this is how RECOMMEND_Q was picked.
import fs from "node:fs";
import { loadLayaNode } from "./laya-node.mjs";
import { judgeBatch } from "../web/laya-batch.js";
import { cleanComment, judgeState, RECOMMEND_Q } from "../web/text.js";
import { auroc } from "../web/metrics.js";

export const VARIANTS = {
  choice2: { pick: (r) => r.probabilities.recommends, q: RECOMMEND_Q },
  choice3: { pick: (r) => r.probabilities.recommends, q: { type: "choice", instructions: "Why does this Hacker News comment link a YouTube video?",
    criteria: { recommends: "To recommend it: the commenter thinks the video is worth watching (a talk, lecture, documentary, explainer, tutorial)",
      evidence: "As evidence, a source or an example in an argument",
      other: "A joke, meme, music, news clip, livestream, bare link, or the commenter's own product" } } },
  noul: { pick: (r) => r.probabilities.true, q: { type: "noul", instructions: "The commenter is recommending the linked video as worth watching" } },
  choice2short: { pick: (r) => r.probabilities.yes, q: { type: "choice", instructions: "Is this comment a recommendation to watch the video?", criteria: { yes: "Yes, a recommendation to watch it", no: "No" } } },
};
const lab = new Map(JSON.parse(fs.readFileSync("data/labels.json", "utf8")).map((x) => [x.id, x.label === "recommends"]));
const rows = fs.readFileSync("data/sample.jsonl", "utf8").trim().split("\n").map(JSON.parse).map((c) => ({ ...c, clean: cleanComment(c.text) }));
rows.sort((a, b) => a.clean.length - b.clean.length);
const L = await loadLayaNode(2);
const out = {};
for (const name of process.argv.slice(2)) {
  const V = VARIANTS[name], ps = [];
  const t0 = Date.now();
  for (let i = 0; i < rows.length; i += 8) {
    const b = rows.slice(i, i + 8);
    const { results } = await judgeBatch(L.ort, L.session, L.tok, L.cfg, b.map(judgeState), V.q);
    results.forEach((r, k) => ps.push([b[k].id, V.pick(r)]));
  }
  out[name] = Object.fromEntries(ps);
  console.log(name, "AUROC", auroc(ps.map(([id, p]) => [p, lab.get(id)])).toFixed(3), `${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
fs.writeFileSync("data/out/laya_variants.json", JSON.stringify(out));
