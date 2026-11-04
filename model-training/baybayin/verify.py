"""Second check: an OpenAI vision model reviews our model's reading.

Our model finds the Baybayin and drafts the reading. The vision model then
looks at the same image and the draft, adds Baybayin words that were missed,
fixes misread syllables, drops Latin text that was read by mistake, and gives
a natural English translation. Without an API key (or if the call fails) the
draft is returned unchanged with a note saying the check did not run.

Config (netlify/.env or the project-root .env, both git-ignored, or environment variables):
    OPENAI_API_KEY   required for the check
    OPENAI_MODEL     vision-capable model id (default luna)
The key stays on the local server; it is never sent to the browser.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "luna"
DEFAULT_EFFORT = "low"
MAX_SIDE = 1600
TIMEOUT = 90

PROMPT = """You verify Baybayin OCR. You get a photo and a DRAFT from a Baybayin OCR model.

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
}"""


ENV_FILES = [ROOT / "netlify" / ".env", ROOT / ".env"]   # git-ignored; first value found wins


def load_env() -> None:
    """Read KEY=VALUE lines from netlify/.env (then the project-root .env) into
    os.environ, without overriding variables that are already set."""
    for path in ENV_FILES:
        if path.exists():
            _read_env(path)


def _read_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config() -> tuple[str | None, str]:
    load_env()
    key = os.environ.get("OPENAI_API_KEY") or None
    if key and key.startswith("replace-with"):
        key = None
    return key, os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def _image_data_url(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def _draft(result: dict) -> dict:
    """The parts of our model's result the reviewer needs (no overlay image)."""
    return {
        "baybayin": result.get("baybayin", ""),
        "tagalog": result.get("tagalog", ""),
        "english_word_by_word": result.get("english", ""),
        "words": [{"line": w["line"], "baybayin": w["baybayin"], "reading": w["reading"],
                   "tagalog": w["tagalog"], "confidence": round(w["confidence"], 2), "box_xyxy": w["box"]}
                  for w in result.get("words", [])],
    }


def _post(body: dict, key: str) -> dict:
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify(img: np.ndarray, result: dict) -> dict:
    """Return {"status": "ok"|"unavailable"|"error", "model", "message", "verified"?}."""
    key, model = config()
    if not key:
        return {"status": "unavailable", "model": model,
                "message": "AI check is off: add OPENAI_API_KEY to netlify/.env."}
    body = {
        "model": model,   # Luna models only accept their default temperature
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "DRAFT:\n" + json.dumps(_draft(result), ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": _image_data_url(img), "detail": "high"}},
            ]},
        ],
    }
    effort = os.environ.get("OPENAI_REASONING_EFFORT", DEFAULT_EFFORT)
    if effort:
        body["reasoning_effort"] = effort   # less "thinking" time before answering
    try:
        try:
            data = _post(body, key)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "reasoning_effort" in body:   # model without this setting
                body.pop("reasoning_effort")
                data = _post(body, key)
            else:
                raise
        content = data["choices"][0]["message"]["content"]
        out = json.loads(content)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        return {"status": "error", "model": model, "message": f"AI check failed ({e.code}): {detail}"[:300]}
    except Exception as e:  # network, timeout, bad JSON
        return {"status": "error", "model": model, "message": f"AI check failed: {e}"[:300]}
    verified = {
        "has_baybayin": bool(out.get("has_baybayin", True)),
        "baybayin": str(out.get("baybayin", "")),
        "tagalog": str(out.get("tagalog", "")),
        "english": str(out.get("english", "")),
        "all_symbols_detected": bool(out.get("all_symbols_detected", False)),
        "changes": [str(c) for c in out.get("changes", [])][:20],
        "confidence": str(out.get("confidence", "")),
        "note": str(out.get("note", "")),
    }
    return {"status": "ok", "model": model, "message": "", "verified": verified}
