# Baybayin AI (BaybAI)

Web app that reads Baybayin from a photo, transliterates it to Tagalog and
translates it to English. Everything runs on **Netlify**: the website and two
JavaScript functions.

```
browser ── Netlify
            ├── website (index.html, css/, js/) + Firebase (chat history)
            ├── /api/ocr                       netlify/functions/ocr.mjs
            │     1. trained model (paper method, netlify/lib/baybai.mjs):
            │        preprocessing → CNN / diacritic CNN / script SVM → Tagalog word list
            │     2. OpenAI Luna review (netlify/lib/luna.mjs): corrections + English
            └── /.netlify/functions/translate  netlify/functions/translate.mjs (Translate button)
```

The models are trained in Python (`model-training/`) and exported for the
function with `python export_web_models.py` (→ `netlify/ocr-data/`).

## Structure
- `index.html`, `css/`, `js/` — website
- `database/` — Firebase web config and Firestore rules
- `netlify/functions/` — `ocr.mjs` (Baybayin OCR) and `translate.mjs` (text translation)
- `netlify/lib/` — `baybai.mjs` (the paper pipeline in JavaScript), `luna.mjs` (Luna review)
- `netlify/ocr-data/` — exported trained models, word list, frequencies, dictionary (~10 MB)
- `netlify/tests/parity.mjs` — checks the JavaScript pipeline against the Python results
- `netlify/build.mjs`, `netlify.toml` — Netlify build (publishes only the website files)
- `model-training/` — training, evaluation and export (Python; see its README)
- `assets/` — logo/favicon and the datasets

## Run locally
Requires Node.js 20+ and the Netlify CLI (`npm install -g netlify-cli`; set
`npm_config_cache` to a folder on D: first if C: is full).
```powershell
npm install          # installs sharp (image decoding)
netlify dev          # website + functions at http://localhost:8888
```
Keys are read from `netlify/.env` (git-ignored):
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-6-luna
```

## Deploy on Netlify
1. Import the repository into Netlify (base directory empty). `netlify.toml`
   sets the build command, publish folder (`dist`), functions and the bundled
   model files.
2. Site configuration → Environment variables:
   | Name | Value |
   |---|---|
   | `OPENAI_API_KEY` | your OpenAI key |
   | `OPENAI_MODEL` | `gpt-6-luna` (or `gpt-5.6-luna`) |
3. Deploy. Only the website files and the two functions are published; the
   datasets and the Python training code are not.
4. Firebase console → Authentication → Settings → Authorized domains → add your
   Netlify domain. Publish `database/firestore.rules` if not done already.

Limits: Netlify functions get 6 MB per request (the website shrinks photos to
1600 px before upload) and 60 s per call (a photo with the Luna review takes
about 8–25 s).

## Security
- The OpenAI key lives only in `netlify/.env` locally and in Netlify's
  environment settings; it is never sent to the browser or committed.
- Uploaded photos are sent to OpenAI for the Luna review (`/api/ocr?verify=0`
  skips it).
