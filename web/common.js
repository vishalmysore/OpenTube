// Shared helpers for every page: data loading, DOM helpers, YouTube titles, formatting.
import { TOPICS } from "./topics.js";

export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const pct = (p, d = 0) => (p == null || !Number.isFinite(p) ? "–" : `${(p * 100).toFixed(d)}%`);
export const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
export const TOPIC = Object.fromEntries(TOPICS.map(([id, name]) => [id, name]));
export const thumb = (v) => `https://i.ytimg.com/vi/${v}/mqdefault.jpg`;
export const watch = (v) => `https://www.youtube.com/watch?v=${v}`;
export const hnItem = (id) => `https://news.ycombinator.com/item?id=${id}`;
export const hnUser = (u) => `https://news.ycombinator.com/user?id=${encodeURIComponent(u)}`;
export const yieldToBrowser = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });

export function duration(sec) {
  if (sec < 90) return `${sec.toFixed(0)} s`;
  if (sec < 5400) return `${(sec / 60).toFixed(0)} min`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)} h`;
  return `${(sec / 86400).toFixed(1)} days`;
}

const cache = {};
export async function load(name) {
  if (!cache[name]) cache[name] = fetch(`./data/${name}`).then((r) => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return name.endsWith(".json") ? r.json() : r.arrayBuffer(); });
  return cache[name];
}

// Video titles are not in the HN data. They are looked up in the viewer's browser through noembed.com (CORS-enabled
// oEmbed proxy) only for cards on screen, and remembered per browser. If the lookup fails, the card falls back to
// the thread title and the recommendation text.
const TKEY = "otg.titles.v1";
let titles = {};
try { titles = JSON.parse(localStorage.getItem(TKEY) || "{}"); } catch { titles = {}; }
let saveTimer = null;
const persist = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => { try { localStorage.setItem(TKEY, JSON.stringify(titles)); } catch { /* full or blocked */ } }, 400); };
const inflight = new Map();
let active = 0; const waiters = [];
const slot = async () => { if (active < 6) { active++; return; } await new Promise((r) => waiters.push(r)); active++; };
const free = () => { active--; waiters.shift()?.(); };

export function cachedTitle(v) { return titles[v] || null; }
export async function videoTitle(v) {
  if (titles[v]) return titles[v];
  if (inflight.has(v)) return inflight.get(v);
  const p = (async () => {
    await slot();
    try {
      const r = await fetch(`https://noembed.com/embed?url=${encodeURIComponent(watch(v))}`);
      const j = await r.json();
      if (j && j.title) { titles[v] = { t: j.title, a: j.author_name || "" }; persist(); return titles[v]; }
      if (j && j.error) { titles[v] = { t: null, a: "", gone: true }; persist(); }
    } catch { /* offline or blocked */ } finally { free(); }
    return null;
  })();
  inflight.set(v, p);
  return p;
}

export function nav(current) {
  const pages = [["index.html", "Watch"], ["ask.html", "Ask the graph"], ["receipts.html", "Receipts"], ["label.html", "Labels"]];
  return `<header class="top">
  <a class="brand" href="./index.html"><span class="logo"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M8 5.5v13l11-6.5z" fill="#fff"/></svg></span><span><b>OpenTube Graph</b><br><span>Hacker News's video canon, filtered in the open</span></span></a>
  <nav class="nav" aria-label="Pages">${pages.map(([h, l]) => `<a href="./${h}"${h === current ? ' aria-current="page"' : ""}>${l}</a>`).join("")}</nav>
</header>`;
}

export function footer() {
  return `<footer class="foot">Data: <a href="https://huggingface.co/datasets/open-index/hacker-news">open-index/hacker-news</a> (ODC-By). Stage 1: <a href="https://huggingface.co/VishalMysore/layaForWebTrained">Laya typed-decisions</a> (ONNX). Stage 2: Qwen2.5-1.5B-Instruct (<a href="https://github.com/mlc-ai/web-llm">WebLLM</a> in the browser, ONNX offline). Retrieval: all-MiniLM-L6-v2 + SAGE-style graph expansion. Inspired by Andrej Karpathy's HackerTube; not affiliated with him, Hacker News or YouTube. <a href="https://github.com/vishalmysore/OpenTube">Source</a> · Apache-2.0.</footer>`;
}

export function mountChrome(current) {
  document.body.insertAdjacentHTML("afterbegin", nav(current));
  document.body.insertAdjacentHTML("beforeend", footer());
}
