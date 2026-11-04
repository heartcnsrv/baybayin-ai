// POST /api/ocr  — body: the image bytes (JPEG / PNG / WebP).
// 1. our paper model (netlify/lib/baybai.mjs): characters -> Tagalog (+ dictionary English)
// 2. Luna review (netlify/lib/luna.mjs): corrections + English for phrases/sentences
// Add ?verify=0 to skip the Luna check, ?single=1 for an image of one word.
import sharp from "sharp";
import { recognize, toGray } from "../lib/baybai.mjs";
import { verify } from "../lib/luna.mjs";

const json = (status, obj) => new Response(JSON.stringify(obj), { status,
  headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" } });
const MAX_UPLOAD = 6 * 1024 * 1024;   // Netlify request limit

export default async (req) => {
  if (req.method !== "POST") return json(405, { error: "Method not allowed" });
  const t0 = Date.now();
  const bytes = Buffer.from(await req.arrayBuffer());
  if (!bytes.length || bytes.length > MAX_UPLOAD) return json(413, { error: "Send one image up to 6 MB." });
  const url = new URL(req.url);

  let gray, lunaImage;
  try {
    const img = sharp(bytes).rotate();   // respect phone photo orientation
    const { data, info } = await img.clone().toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
    gray = toGray(data, info.width, info.height, info.channels);
    const jpg = await img.clone().resize({ width: 1600, height: 1600, fit: "inside", withoutEnlargement: true })
      .flatten({ background: "#ffffff" }).jpeg({ quality: 88 }).toBuffer();
    lunaImage = `data:image/jpeg;base64,${jpg.toString("base64")}`;
  } catch {
    return json(400, { error: "That file is not an image the reader can open (use JPG, PNG or WebP)." });
  }

  let res;
  try {
    res = recognize(gray, url.searchParams.get("single") === "1");
  } catch (e) {
    return json(500, { error: `Recognition failed: ${e.message}` });
  }
  res.model_seconds = (Date.now() - t0) / 1000;
  res.draft = { baybayin: res.baybayin, tagalog: res.tagalog, english: res.english };

  let check = { status: "skipped", model: "", message: "AI check skipped." };
  if (url.searchParams.get("verify") !== "0") {
    check = await verify(lunaImage, res);
    // Second look: answers vary between calls. If Luna finds no Baybayin although
    // our model detected some, ask once more (only while there is time left).
    if (check.status === "ok" && !check.verified.has_baybayin && res.words.length && Date.now() - t0 < 25000) {
      const again = await verify(lunaImage, res, 25000);
      if (again.status === "ok") check = again;
    }
  }
  res.verifier = { status: check.status, model: check.model, message: check.message };
  if (check.status === "ok") {
    const v = check.verified;
    if (!v.has_baybayin) Object.assign(res, { baybayin: "", tagalog: "", english: "", words: [] });
    else if (v.tagalog.trim()) Object.assign(res, { baybayin: v.baybayin, tagalog: v.tagalog, english: v.english });
    Object.assign(res.verifier, { has_baybayin: v.has_baybayin, all_symbols_detected: v.all_symbols_detected,
      changes: v.changes, confidence: v.confidence, note: v.note });
  }
  res.seconds = (Date.now() - t0) / 1000;
  return json(200, res);
};

export const config = { path: "/api/ocr" };
