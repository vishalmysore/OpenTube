// WebLLM (MLC) in the page: the stage-2 judge for comments Laya is unsure about, and the answer writer for GraphRAG.
// Loaded the same way as gamWebLLM / recursiveMASWebLLM. Needs WebGPU (Chrome/Edge 113+, recent Safari/Firefox).
import { llmMessages, pYes, LLM_MODEL } from "./llm-prompt.js";

const WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm@0.2.84";
let webllm = null;
export const W = { engine: null, model: null };

export async function hasWebGPU() {
  try { return !!navigator.gpu && !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

export async function loadWebLLM(model = LLM_MODEL.webllm, { onStatus = () => {}, onProgress = () => {} } = {}) {
  if (W.engine && W.model === model) return W.engine;
  if (!(await hasWebGPU())) throw new Error("WebGPU is not available in this browser, so WebLLM cannot run. Try Chrome or Edge on a desktop.");
  webllm ||= await import(WEBLLM_URL);
  onStatus(`Loading ${model}…`);
  W.engine = await webllm.CreateMLCEngine(model, {
    initProgressCallback: (r) => { onProgress(r.progress ?? null); onStatus(r.text || `Loading ${model}…`); },
  });
  W.model = model;
  return W.engine;
}

/** Stage-2 yes/no judgment with the same prompt as the offline run; p(yes) from the first token's log-probs. */
export async function llmJudge(state) {
  const t0 = performance.now();
  const r = await W.engine.chat.completions.create({
    messages: llmMessages(state), max_tokens: 1, temperature: 0, logprobs: true, top_logprobs: 5,
  });
  const ms = performance.now() - t0;
  const top = r.choices[0].logprobs?.content?.[0]?.top_logprobs || [];
  const pick = (re) => { const m = top.filter((t) => re.test(t.token.trim())); return m.length ? Math.log(m.reduce((a, t) => a + Math.exp(t.logprob), 0)) : null; };
  const p = pYes(pick(/^yes$/i), pick(/^no$/i));
  return { p, ms, promptTokens: r.usage?.prompt_tokens ?? null, raw: r.choices[0].message.content };
}

/** Stream an answer. onToken gets the text so far. */
export async function llmAnswer(messages, onToken, { maxTokens = 450 } = {}) {
  const t0 = performance.now();
  const stream = await W.engine.chat.completions.create({ messages, stream: true, temperature: 0.2, max_tokens: maxTokens, stream_options: { include_usage: true } });
  let text = "", usage = null;
  for await (const chunk of stream) {
    text += chunk.choices[0]?.delta?.content || "";
    if (chunk.usage) usage = chunk.usage;
    onToken(text);
  }
  return { text, usage, ms: performance.now() - t0 };
}
