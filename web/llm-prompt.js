// The stage-2 prompt. Shared by the offline run (transformers.js, Node) and the live page (WebLLM), so both ask the
// small LLM exactly the same thing. The answer is read from the first token's yes/no probabilities, not parsed text.
export const LLM_MODEL = { webllm: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", hf: "onnx-community/Qwen2.5-1.5B-Instruct" };

export function llmMessages(c) {
  return [
    { role: "system", content: "You label Hacker News comments. Answer with one word: yes or no." },
    { role: "user", content:
`Thread title: ${c.thread || "(unknown)"}
Comment: ${c.comment}

Is the commenter recommending the linked YouTube video as worth watching (for example a talk, lecture, documentary, tutorial or explainer they think others should see)?
Answer "no" if the link is incidental, only evidence for a claim, a music track, a joke or meme, a news clip, criticism of the video, or self-promotion.
Answer yes or no.` },
  ];
}

/** p(yes) from two log-probabilities (either may be missing). */
export function pYes(lpYes, lpNo) {
  if (lpYes == null && lpNo == null) return 0.5;
  if (lpYes == null) return 1 - Math.exp(lpNo);
  if (lpNo == null) return Math.exp(lpYes);
  const m = Math.max(lpYes, lpNo);
  return Math.exp(lpYes - m) / (Math.exp(lpYes - m) + Math.exp(lpNo - m));
}
