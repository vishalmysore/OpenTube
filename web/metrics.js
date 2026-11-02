// Scoring shared by the build script (Node) and the Receipts page (browser sliders re-score without re-running models).

/** Decide one comment from its stage outputs. th = { lo, hi, llm, laya } */
export function cascadeDecision(r, th) {
  if (r.laya_p >= th.hi) return { recommends: true, route: "laya_accept" };
  if (r.laya_p <= th.lo) return { recommends: false, route: "laya_reject" };
  return { recommends: (r.llm_p ?? 0) >= th.llm, route: "llm" };
}

export function auroc(pairs) {
  const pos = pairs.filter((p) => p[1]).map((p) => p[0]), neg = pairs.filter((p) => !p[1]).map((p) => p[0]);
  if (!pos.length || !neg.length) return null;
  let s = 0;
  for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

/** Wilson 95% interval for a proportion. */
export function wilson(k, n) {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function confusion(preds, golds) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  preds.forEach((p, i) => { const g = golds[i]; if (p && g) tp++; else if (p) fp++; else if (g) fn++; else tn++; });
  const n = preds.length, acc = (tp + tn) / n, prec = tp / Math.max(1, tp + fp), rec = tp / Math.max(1, tp + fn);
  return { n, tp, fp, tn, fn, accuracy: acc, accuracy_ci: wilson(tp + tn, n), precision: prec, recall: rec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0 };
}

/**
 * @param sample [{ gold, laya_p, llm_p, laya_ms, llm_ms, borderline }]
 * @param th { lo, hi, llm, laya }
 */
export function evaluate(sample, th, { excludeBorderline = false } = {}) {
  const S = excludeBorderline ? sample.filter((s) => !s.borderline) : sample;
  const gold = S.map((s) => s.gold === "recommends");
  const laya = confusion(S.map((s) => s.laya_p >= th.laya), gold);
  const llm = confusion(S.map((s) => s.llm_p >= (th.llm_only ?? th.llm)), gold);
  const dec = S.map((s) => cascadeDecision(s, th));
  const casc = confusion(dec.map((d) => d.recommends), gold);
  const toLLM = dec.filter((d) => d.route === "llm").length / S.length;
  const ms = (k) => S.reduce((a, s) => a + s[k], 0) / S.length;
  laya.ms = ms("laya_ms"); llm.ms = ms("llm_ms"); casc.ms = laya.ms + toLLM * llm.ms; casc.to_llm = toLLM;
  laya.auroc = auroc(S.map((s, i) => [s.laya_p, gold[i]]));
  llm.auroc = auroc(S.map((s, i) => [s.llm_p, gold[i]]));
  // How good are Laya's own auto-decisions (the ones that skip the LLM)?
  const auto = S.map((s, i) => [dec[i], gold[i]]).filter(([d]) => d.route !== "llm");
  const autoAcc = auto.length ? auto.filter(([d, g]) => d.recommends === g).length / auto.length : null;
  return {
    n: S.length, positives: gold.filter(Boolean).length,
    summary: { laya, llm, cascade: casc, laya_auto_accuracy: autoAcc, laya_auto_n: auto.length },
  };
}
