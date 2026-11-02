// Batched Laya calls: many (state, question) pairs in ONE session.run, using laya-core's sequence builder.
// laya-core's systemOne() batches several questions about one state; here every comment is its own state, so
// batching across comments is what makes a full-archive pass affordable on a CPU.
import { toInternal, buildSequence, renderOptions, collate, tempBucket, confidenceFromProbs, specialIds } from "./laya-core.js";

const QTYPES = { choice: 0, score: 1, noul: 2 };

/**
 * @param ort        onnxruntime (node or web)
 * @param session    InferenceSession for the Laya graph
 * @param tok        Tokenizer
 * @param cfg        rl_agent_config.json
 * @param states     array of states (objects or strings)
 * @param question   one typed question { type: "choice", instructions, criteria }
 * @returns array of { probabilities, choice, confidence, tokens } in the same order, plus total ms
 */
export async function judgeBatch(ort, session, tok, cfg, states, question) {
  const sp = specialIds(tok);
  const q = toInternal(question);
  const k = renderOptions(q).length;
  const items = states.map((s) => {
    const { ids, markers } = buildSequence(tok, sp, s, q, cfg.max_len ?? 512, cfg.head_max_len ?? 192);
    return { ids, markers, qtype: QTYPES[q.t] };
  });
  const b = collate(items, sp.pad);
  const feeds = {
    input_ids: new ort.Tensor("int64", b.ids, [b.n, b.L]),
    attention_mask: new ort.Tensor("int64", b.att, [b.n, b.L]),
    marker_pos: new ort.Tensor("int64", b.mpos, [b.n, b.kmax]),
    marker_mask: new ort.Tensor("bool", b.mmask, [b.n, b.kmax]),
    qtype: new ort.Tensor("int64", b.qtype, [b.n]),
  };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  const logits = out.logits.data;
  const tScale = cfg.temperature_by_options?.[tempBucket(QTYPES[q.t], k)] ?? cfg.temperature[QTYPES[q.t]];
  const keys = q.t === "choice" ? Object.keys(q.crit) : ["false", "true"];
  const results = items.map((it, r) => {
    const z = keys.map((_, i) => logits[r * b.kmax + i] / Math.max(1e-3, tScale));
    const zmax = Math.max(...z);
    const e = z.map((v) => Math.exp(v - zmax));
    const s = e.reduce((a, v) => a + v, 0);
    const p = e.map((v) => v / s);
    const top = p.indexOf(Math.max(...p));
    return {
      choice: keys[top],
      probabilities: Object.fromEntries(keys.map((kk, i) => [kk, Math.round(p[i] * 1e4) / 1e4])),
      confidence: Math.round(confidenceFromProbs(p, k) * 1e4) / 1e4,
      tokens: it.ids.length,
    };
  });
  return { results, ms };
}
