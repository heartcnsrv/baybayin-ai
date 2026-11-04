// Secondary check by the OpenAI "Luna" model (paper Section 5), ported from
// model-training/baybayin/verify.py. It reviews the OCR draft against the photo,
// adds missed Baybayin, fixes misreadings, drops non-Baybayin text, and gives the
// English translation (used for phrases and sentences). The API key is read
// from the server environment and never reaches the browser.
const API_URL = "https://api.openai.com/v1/chat/completions";

const PROMPT = `You verify Baybayin OCR. You get a photo and a DRAFT from a Baybayin OCR model.

Task:
1. Read ONLY the Baybayin script in the photo. Latin-letter text (captions, signatures, labels) is NOT Baybayin;
   never transcribe it as Baybayin. You may use Latin captions only as a hint, and say so in "note".
2. Compare with the DRAFT: add Baybayin words it missed, fix wrong syllables (kudlit above = e/i,
   below = o/u, cross/virama = no vowel), remove draft words that are not Baybayin, and fix word breaks.
3. Give the Tagalog transliteration in reading order, one line per line of Baybayin, and a natural,
   accurate English translation of the whole text.
4. Do not invent text that is not visible. Never keep draft words that are not visible Baybayin.
   If a Baybayin word is visible but unclear, keep the draft reading for it and lower "confidence".
5. Baybayin may be handwritten or printed, in any colour (including white lettering over a
   photo), large or small, in a decorative font. Look at the whole image carefully.
   Only if the photo contains NO Baybayin script at all (e.g. only Latin text, drawings or objects),
   set "has_baybayin" to false and return empty strings for "baybayin", "tagalog" and "english".

Return ONLY a JSON object:
{
  "has_baybayin": true or false,
  "baybayin": "Baybayin Unicode, lines separated by \\n",
  "tagalog": "Tagalog transliteration, lines separated by \\n",
  "english": "natural English translation",
  "all_symbols_detected": true or false (did the DRAFT find every Baybayin word?),
  "changes": ["short description of each correction made to the draft"],
  "confidence": "high" | "medium" | "low",
  "note": "one short sentence on anything uncertain"
}`;

export function lunaConfig() {
  const key = process.env.OPENAI_API_KEY || "";
  return { key: key && !key.startsWith("replace-with") ? key : "", model: process.env.OPENAI_MODEL || "gpt-6-luna",
    effort: process.env.OPENAI_REASONING_EFFORT ?? "low" };
}

function draftOf(result) {
  return { baybayin: result.baybayin, tagalog: result.tagalog, english_word_by_word: result.english,
    words: result.words.map((w) => ({ line: w.line, baybayin: w.baybayin, reading: w.reading, tagalog: w.tagalog,
      confidence: Math.round(w.confidence * 100) / 100, box_xyxy: w.box })) };
}

async function post(body, key, timeoutMs) {
  const r = await fetch(API_URL, { method: "POST", signal: AbortSignal.timeout(timeoutMs),
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, data };
}

/** imageDataUrl: JPEG data URL of the photo (max side 1600). Returns {status, model, message, verified?}. */
export async function verify(imageDataUrl, result, timeoutMs = 45000) {
  const { key, model, effort } = lunaConfig();
  if (!key) return { status: "unavailable", model, message: "AI check is off: OPENAI_API_KEY is not set." };
  const body = { model, response_format: { type: "json_object" }, messages: [
    { role: "system", content: PROMPT },
    { role: "user", content: [
      { type: "text", text: "DRAFT:\n" + JSON.stringify(draftOf(result)) },
      { type: "image_url", image_url: { url: imageDataUrl, detail: "high" } }] }] };
  if (effort) body.reasoning_effort = effort;   // less "thinking" time before answering
  try {
    let r = await post(body, key, timeoutMs);
    if (!r.ok && r.status === 400 && body.reasoning_effort) {   // model without this setting
      delete body.reasoning_effort;
      r = await post(body, key, timeoutMs);
    }
    if (!r.ok) return { status: "error", model, message: `AI check failed (${r.status}): ${r.data?.error?.message || ""}`.slice(0, 300) };
    const out = JSON.parse(r.data.choices?.[0]?.message?.content || "{}");
    return { status: "ok", model, message: "", verified: {
      has_baybayin: out.has_baybayin !== false, baybayin: String(out.baybayin || ""), tagalog: String(out.tagalog || ""),
      english: String(out.english || ""), all_symbols_detected: Boolean(out.all_symbols_detected),
      changes: (out.changes || []).map(String).slice(0, 20), confidence: String(out.confidence || ""), note: String(out.note || "") } };
  } catch (e) {
    return { status: "error", model, message: `AI check failed: ${e.message}`.slice(0, 300) };
  }
}
