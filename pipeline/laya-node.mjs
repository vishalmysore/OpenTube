import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";
export const MODEL_DIR = new URL("../models/laya/", import.meta.url).pathname;
export async function loadLayaNode(threads = 2) {
  const j = (f) => JSON.parse(fs.readFileSync(MODEL_DIR + f, "utf8"));
  const tok = new Tokenizer(j("tokenizer.json"), j("tokenizer_config.json"));
  const cfg = j("rl_agent_config.json");
  const session = await ort.InferenceSession.create(MODEL_DIR + (process.env.LAYA_VARIANT||"laya_q8e8")+".onnx", { intraOpNumThreads: threads, graphOptimizationLevel: "all" });
  return { ort, session, tok, cfg };
}
