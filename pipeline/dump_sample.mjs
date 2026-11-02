import fs from "node:fs";
import { cleanComment, window } from "../web/text.js";
const rows = fs.readFileSync("data/sample.jsonl","utf8").trim().split("\n").map(JSON.parse);
const [a,b] = process.argv.slice(2).map(Number);
rows.slice(a,b).forEach((c,i)=>console.log(`#${a+i} ${c.id} | ${c.story_title.slice(0,70)}\n  ${window(cleanComment(c.text),420).replace(/\n/g," / ")}`));
