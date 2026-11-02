// Shared text helpers (Node pipeline and browser pages use the same code, so the live demo sees exactly what the
// offline run saw).

const ENT = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'", "&#39;": "'", "&#x2F;": "/", "&#47;": "/", "&nbsp;": " " };

/** HN comment HTML -> plain text. YouTube links become "[YouTube video]" so the model judges the words, not the URL. */
export function cleanComment(html) {
  let s = String(html || "");
  s = s.replace(/<a\s[^>]*href="([^"]*)"[^>]*>.*?<\/a>/gis, (_, href) => (/youtu/i.test(href) ? " [YouTube video] " : " [link] "));
  s = s.replace(/<p>/gi, "\n").replace(/<\/?(i|b|em|strong|code|pre)>/gi, "").replace(/<[^>]+>/g, "");
  s = s.replace(/&(?:amp|lt|gt|quot|nbsp);|&#x27;|&#39;|&#x2F;|&#47;/g, (m) => ENT[m] ?? m).replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n)).replace(/&#x([0-9a-f]+);/gi, (_, n) => String.fromCodePoint(parseInt(n, 16)));
  s = s.replace(/https?:\/\/(?:www\.|m\.)?(?:youtube\.com|youtu\.be)\/\S+/gi, "[YouTube video]");
  return s.replace(/[ \t]+/g, " ").replace(/\n\s*/g, "\n").trim();
}

/** Trim long comments around the first video mention so the judge sees the sentence that matters. */
export function window(text, max = 900) {
  if (text.length <= max) return text;
  const i = Math.max(0, text.indexOf("[YouTube video]"));
  const start = Math.max(0, Math.min(i - Math.floor(max * 0.6), text.length - max));
  return (start > 0 ? "… " : "") + text.slice(start, start + max) + (start + max < text.length ? " …" : "");
}

/** The state Laya sees for one comment. */
export function judgeState(c) {
  return { thread: c.story_title || "", comment: window(c.clean) };
}

/** The typed question for the cheap pass. Kept here so the page and the pipeline can't drift apart. */
export const RECOMMEND_Q = {
  type: "choice",
  instructions: "Is the commenter recommending the linked YouTube video as worth watching?",
  criteria: {
    recommends: "Yes: the comment points people to the video because it is good, useful, interesting or worth their time",
    not_recommending: "No: the link is incidental, a citation for a claim, a joke, music, a news clip, criticism of the video, or self-promotion",
  },
};
