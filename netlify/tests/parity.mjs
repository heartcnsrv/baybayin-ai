// Checks that the JavaScript port (netlify/lib/baybai.mjs) gives the same results
// as the Python pipeline on the held-out word images.
//   cd model-training && python evaluate.py test     (writes runs/word_test.json)
//   node netlify/tests/parity.mjs
import { readFileSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";
import { recognize, toGray } from "../lib/baybai.mjs";

const root = join(import.meta.dirname, "..", "..");
const ref = JSON.parse(readFileSync(join(root, "model-training", "runs", "word_test.json"), "utf8")).per_image;
const limit = Number(process.argv[2] || ref.length);

let sameTranslit = 0, sameTagalog = 0, jsCorrect = 0, pyCorrect = 0, n = 0;
const diffs = [];
const t0 = Date.now();
for (const row of ref.slice(0, limit)) {
  const file = join(root, "assets", "Baybayin Word Images", row.file);
  const { data, info } = await sharp(readFileSync(file)).toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
  const out = recognize(toGray(data, info.width, info.height, info.channels));
  const translit = out.transliteration.split("\n").join(" "), tagalog = out.tagalog.split("\n").join(" ");
  n++;
  sameTranslit += translit === row.transliteration;
  sameTagalog += tagalog === row.tagalog;
  jsCorrect += tagalog === row.reference;
  pyCorrect += row.tagalog === row.reference;
  if (tagalog !== row.tagalog && diffs.length < 15) diffs.push(`${row.file}: python "${row.tagalog}" / js "${tagalog}"`);
}
const pct = (v) => `${(100 * v / n).toFixed(2)}%`;
console.log(`images ${n}  (${((Date.now() - t0) / n).toFixed(0)} ms per image)`);
console.log(`same transliteration as Python: ${pct(sameTranslit)}`);
console.log(`same Tagalog word as Python:    ${pct(sameTagalog)}`);
console.log(`Tagalog identification  Python ${pct(pyCorrect)}  JS ${pct(jsCorrect)}`);
if (diffs.length) console.log("differences:\n  " + diffs.join("\n  "));
