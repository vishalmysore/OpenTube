import { test } from "node:test";
import assert from "node:assert/strict";
import { cleanComment, window, judgeState } from "../web/text.js";
import { cascadeDecision, evaluate, auroc, wilson } from "../web/metrics.js";
import { sageRetrieve, buildAdjacency, scoreAll } from "../web/sage.js";
import { pYes } from "../web/llm-prompt.js";

test("cleanComment replaces YouTube links and unescapes HN HTML", () => {
  const html = 'Great talk: <a href="https:&#x2F;&#x2F;www.youtube.com&#x2F;watch?v=abcdefghijk" rel="nofollow">https:&#x2F;&#x2F;www.youtube.com&#x2F;watch?v=abcdefghijk</a><p>It&#x27;s &quot;good&quot;';
  assert.equal(cleanComment(html), 'Great talk: [YouTube video] \nIt\'s "good"');
  assert.match(cleanComment('see <a href="https://example.com">x</a>'), /\[link\]/);
});

test("window keeps the video mention in long comments", () => {
  const t = "a".repeat(2000) + " [YouTube video] " + "b".repeat(2000);
  const w = window(t, 900);
  assert.ok(w.includes("[YouTube video]"));
  assert.ok(w.length <= 910);
  assert.deepEqual(Object.keys(judgeState({ story_title: "T", clean: "c" })), ["thread", "comment"]);
});

test("cascade routes by thresholds", () => {
  const th = { lo: 0.4, hi: 0.7, llm: 0.5 };
  assert.deepEqual(cascadeDecision({ laya_p: 0.8 }, th), { recommends: true, route: "laya_accept" });
  assert.deepEqual(cascadeDecision({ laya_p: 0.1 }, th), { recommends: false, route: "laya_reject" });
  assert.deepEqual(cascadeDecision({ laya_p: 0.5, llm_p: 0.9 }, th), { recommends: true, route: "llm" });
});

test("evaluate computes accuracy and LLM share", () => {
  const S = [
    { gold: "recommends", laya_p: 0.9, llm_p: 0.9, laya_ms: 1, llm_ms: 3 },
    { gold: "not_recommending", laya_p: 0.1, llm_p: 0.1, laya_ms: 1, llm_ms: 3 },
    { gold: "recommends", laya_p: 0.5, llm_p: 0.8, laya_ms: 1, llm_ms: 3 },
    { gold: "not_recommending", laya_p: 0.5, llm_p: 0.7, laya_ms: 1, llm_ms: 3 },
  ];
  const A = evaluate(S, { lo: 0.4, hi: 0.7, llm: 0.75, laya: 0.5 });
  assert.equal(A.summary.cascade.accuracy, 1);
  assert.equal(A.summary.cascade.to_llm, 0.5);
  assert.equal(A.summary.cascade.ms, 1 + 0.5 * 3);
  assert.equal(auroc([[0.9, true], [0.1, false]]), 1);
  const [lo, hi] = wilson(50, 100); assert.ok(lo < 0.5 && hi > 0.5);
});

test("SAGE expansion pulls in a structurally linked neighbour", () => {
  const D = 2, mat = new Int8Array([127, 0, 0, 127, 90, 90]);
  const scores = scoreAll(Float32Array.from([1, 0]), mat, D);
  const adj = buildAdjacency(3, [[0, 1, "thread", 1, [1, "Ask HN"]]]);
  const r = sageRetrieve(scores, adj, { seeds: 1, expand: 2 });
  assert.equal(r.seeds[0].i, 0);
  assert.equal(r.expanded[0].i, 1);
  assert.equal(r.expanded[0].kind, "thread");
});

test("pYes normalizes the two log-probabilities", () => {
  assert.ok(Math.abs(pYes(Math.log(0.3), Math.log(0.1)) - 0.75) < 1e-9);
  assert.equal(pYes(null, null), 0.5);
});
