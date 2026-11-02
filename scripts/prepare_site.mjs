// Assemble the static site into dist/: the pages and data from web/ plus vendored ONNX Runtime Web and Tokenizers.js
// and the license files. No model is bundled: Laya streams from Hugging Face (VishalMysore/layaForWebTrained), the
// embedder from Xenova/all-MiniLM-L6-v2, and WebLLM fetches Qwen2.5-1.5B-Instruct from MLC's Hugging Face repo.
//   npm ci && node scripts/prepare_site.mjs [--out dist]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arg = (n) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : undefined; };
const dist = path.resolve(root, arg("--out") || "dist");
const nm = path.join(root, "node_modules");
const need = (p, hint) => { if (!fs.existsSync(p)) { console.error(`Missing ${path.relative(root, p)}. ${hint}`); process.exit(1); } };

need(nm, "Install dependencies first: npm ci");
need(path.join(root, "web/data/videos.json"), "Build the data first (see README: pipeline).");
fs.mkdirSync(dist, { recursive: true });
for (const e of fs.readdirSync(dist)) fs.rmSync(path.join(dist, e), { recursive: true, force: true });
fs.cpSync(path.join(root, "web"), dist, { recursive: true });
fs.mkdirSync(path.join(dist, "vendor"), { recursive: true });
const vendor = [
  ["onnxruntime-web/dist/ort.min.mjs", "ort.min.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.jsep.wasm"],
  ["@huggingface/tokenizers/dist/tokenizers.min.mjs", "tokenizers.min.mjs"],
];
for (const [src, dst] of vendor) { const p = path.join(nm, src); need(p, "Check the pinned versions in package.json."); fs.copyFileSync(p, path.join(dist, "vendor", dst)); }
for (const f of ["NOTICE.md", "LICENSE"]) { need(path.join(root, f), "It must be in the repository root."); fs.copyFileSync(path.join(root, f), path.join(dist, f)); }
fs.cpSync(path.join(root, "licenses"), path.join(dist, "licenses"), { recursive: true });
fs.writeFileSync(path.join(dist, ".nojekyll"), "");
let total = 0;
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).forEach((e) => { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else total += fs.statSync(p).size; });
walk(dist);
console.log(`dist/ ready: ${(total / 1e6).toFixed(1)} MB`);
