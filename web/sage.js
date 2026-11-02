// GraphRAG retrieval over the video graph, following SAGE (Structure Aware Graph Expansion, arXiv:2602.16964) as
// implemented in sagejava (github.com/vishalmysore/sagejava): offline percentile-pruned similarity edges, then at
// query time (1) seed selection by dense similarity and (2) neighbour expansion along graph edges.
//
// This graph adds two structural edge kinds the paper's chunk graph does not have, because HN gives them for free:
//   thread    two videos recommended in the same HN thread
//   person    two videos recommended by the same commenter
// Pure functions: no DOM, no model. Unit-tested in tests/sage.test.mjs.

export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/** Dequantize row i of an int8 matrix (scale 1/127) into a Float32Array. */
export function row(mat, dim, i) { const v = new Float32Array(dim); for (let d = 0; d < dim; d++) v[d] = mat[i * dim + d] / 127; return v; }

/** Cosine of a unit query against every int8 row (rows were L2-normalized before quantizing). */
export function scoreAll(q, mat, dim) {
  const n = mat.length / dim, out = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; const o = i * dim; for (let d = 0; d < dim; d++) s += q[d] * mat[o + d]; out[i] = s / 127; }
  return out;
}

export function topK(scores, k, filter = () => true) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (filter(i)) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, k);
}

const EDGE_WEIGHT = { sim: 1.0, thread: 0.8, person: 0.6 };

/**
 * @param scores    Float32Array query similarity per node
 * @param adj       adjacency: adj[i] = [{ j, kind: "sim"|"thread"|"person", w, via }]
 * @param opts      { seeds = 4, expand = 6, minSeed = 0.2, popularity: Float32Array (0..1), alpha = 0.15 }
 * @returns { seeds: [{i, score}], expanded: [{i, score, from, kind, via}] }
 */
export function sageRetrieve(scores, adj, opts = {}) {
  const { seeds: S = 4, expand: E = 6, minSeed = 0.2, popularity = null, alpha = 0.15 } = opts;
  const rank = (i) => scores[i] + (popularity ? alpha * popularity[i] : 0);
  const order = Array.from(scores.keys()).sort((a, b) => rank(b) - rank(a));
  const seeds = order.slice(0, S).filter((i, k) => k === 0 || scores[i] >= minSeed).map((i) => ({ i, score: rank(i) }));
  const inSeed = new Set(seeds.map((s) => s.i));
  const best = new Map();
  for (const s of seeds) {
    for (const e of adj[s.i] || []) {
      if (inSeed.has(e.j)) continue;
      // A neighbour is worth bringing in when it is related to the query AND strongly tied to a seed.
      const sc = (0.5 * scores[e.j] + 0.5 * s.score) * (EDGE_WEIGHT[e.kind] ?? 0.5) * (e.w ?? 1);
      const cur = best.get(e.j);
      if (!cur || sc > cur.score) best.set(e.j, { i: e.j, score: sc, from: s.i, kind: e.kind, via: e.via });
    }
  }
  const expanded = [...best.values()].sort((a, b) => b.score - a.score).slice(0, E);
  return { seeds, expanded };
}

/** Build adjacency lists from the compact edge arrays shipped in graph.json. */
export function buildAdjacency(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [a, b, kind, w, via] of edges) {
    adj[a].push({ j: b, kind, w, via });
    adj[b].push({ j: a, kind, w, via });
  }
  return adj;
}
