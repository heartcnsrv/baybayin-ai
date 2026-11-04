"""Local test server: the existing web UI + the paper's Baybayin OCR pipeline
(translate_pipeline.py), followed by the secondary AI check (baybayin/verify.py).

    python serve.py            # then open http://localhost:8000

Serves the site from the project root and adds POST /api/ocr, which takes the
raw image bytes as the request body and returns JSON (baybayin, tagalog,
english, per-word details; with ?debug=1 also the detection-box image as a data URL).
Only binds to localhost.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from baybayin import verify
from concurrent.futures import ThreadPoolExecutor

from translate_pipeline import BaybayinTranslator, build_translator, english_for, load_params, recognize

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
MAX_UPLOAD = 15 * 1024 * 1024
# Only the web app's own files are served; never .env, .git, data or models.
ALLOWED = [SITE / "index.html", SITE / "css", SITE / "js", SITE / "database" / "firebase-config.js",
           SITE / "assets" / "logo.svg", SITE / "assets" / "favicon.svg"]

translator: BaybayinTranslator | None = None
lock = threading.Lock()      # torch inference is not re-entrant-safe across threads
nmt_lock = threading.Lock()  # one translation at a time


def _allowed(path: Path) -> bool:
    path = path.resolve()
    return any(path == a.resolve() or a.resolve() in path.parents for a in ALLOWED if a.exists())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        rel = urlparse(self.path).path.lstrip("/") or "index.html"
        target = (SITE / rel)
        if not target.is_file() or not _allowed(target):
            self._send(404, b"Not found", "text/plain")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or target.suffix in (".js", ".svg"):
            ctype += "; charset=utf-8"
        if target.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/ocr":
            self._json(404, {"error": "Not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            self._json(413 if n > MAX_UPLOAD else 400, {"error": "Send one image up to 15 MB."})
            return
        data = np.frombuffer(self.rfile.read(n), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            self._json(400, {"error": "That file is not an image the reader can open (use JPG or PNG)."})
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        t = time.time()
        try:
            with lock:   # characters + Tagalog only; English is done below
                res = recognize(translator, img, load_params(), do_english=False)
        except Exception as e:  # report instead of dropping the connection
            self._json(500, {"error": f"Recognition failed: {e}"})
            return
        # The detection boxes are for development only (?debug=1); the website
        # shows users their own photo without boxes.
        overlay = res.pop("overlay")
        if "debug=1" in (urlparse(self.path).query or ""):
            ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
            res["overlay"] = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else ""
        res["model_seconds"] = round(time.time() - t, 2)

        # Step 2, in parallel: our English (dictionary / NLLB) and the AI review
        # of the draft. The AI result is used when it succeeds, ours otherwise.
        skip_check = "verify=0" in (urlparse(self.path).query or "")
        with ThreadPoolExecutor(2) as pool:
            ours = pool.submit(self._english, res["tagalog"])
            check = None if skip_check else pool.submit(verify.verify, img, res)
            res["english"] = ours.result()
            check = check.result() if check else {"status": "skipped", "model": "", "message": "AI check skipped."}
        # Second look: the AI check's answers vary between calls. If it finds no
        # Baybayin although our model detected some, ask once more.
        if (check["status"] == "ok" and not check["verified"]["has_baybayin"] and res.get("words")):
            again = verify.verify(img, res)
            if again["status"] == "ok":
                check = again
        res["draft"] = {k: res[k] for k in ("baybayin", "tagalog", "english")}
        res["verifier"] = {k: v for k, v in check.items() if k != "verified"}
        if check["status"] == "ok":
            v = check["verified"]
            if not v["has_baybayin"]:   # the AI check found no Baybayin in the photo
                res.update({"baybayin": "", "tagalog": "", "english": "", "words": []})
            elif v["tagalog"].strip():
                res.update({"baybayin": v["baybayin"], "tagalog": v["tagalog"], "english": v["english"]})
            res["verifier"].update({k: v[k] for k in ("has_baybayin", "all_symbols_detected", "changes",
                                                      "confidence", "note")})
        res["seconds"] = round(time.time() - t, 2)
        self._json(200, res)

    @staticmethod
    def _english(tagalog: str) -> str:
        with nmt_lock:   # the translation model is shared between requests
            return english_for(translator, tagalog)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    global translator
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print("Loading models and dictionaries...", flush=True)
    translator = build_translator(use_nmt=True)
    translator.translator._load_nmt()   # load NLLB now, not on the first upload
    key, model = verify.config()
    print(f"AI check: {'on, model ' + model if key else 'off (no OPENAI_API_KEY in .env)'}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
