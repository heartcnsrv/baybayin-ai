# Baybayin AI — Netlify web app

Plain HTML, CSS and JavaScript recreation of the supplied UI. Firebase configuration and Firestore rules live under `database/`; translation history is stored per Firebase user in Firestore. AI requests go through a Netlify Function so the OpenAI secret stays on the backend and is not exposed to browser JavaScript.

## Structure
- `index.html` — app structure
- `css/styles.css` — responsive styling
- `js/app.js` — keyboard, translation UI, Firebase auth/database
- `assets/` — logo and favicon
- `netlify/functions/translate.mjs` — server-side OpenAI proxy
- `netlify/functions/ocr.mjs` — secure proxy to the OCR service
- `backend/ocr/` — Python OCR service, preprocessing pipeline, and trained CNN weights
- `netlify/netlify.toml` — Netlify build and function configuration
- `database/firebase-config.js` — public Firebase web-app configuration used by the browser
- `database/firestore.rules` — per-user Firestore access rules
- `netlify/.env.example` — safe local environment-variable template
- `reference-data/model-metadata/` — model class list and findings from supplied OCR ZIP

## Firebase setup
The public Firebase web config is in [`database/firebase-config.js`](database/firebase-config.js). In Firebase Console for `baybayin-ai`, enable **Authentication → Sign-in method → Anonymous and Google**, create/enable Firestore, and publish [`database/firestore.rules`](database/firestore.rules). [`database/firebase.json`](database/firebase.json) points Firebase CLI to those rules; deploy from the repository root with `firebase --config database/firebase.json deploy --only firestore:rules`. Add your Netlify domain under Authentication → Settings → Authorized domains.

Translations live at `users/{uid}/translations/{documentId}` with source, Tagalog transliteration, English translation, and server timestamp. Firebase web config is not a server secret; access control comes from auth and Firestore rules.

## OpenAI key security
`OPENAI_API_KEY` is read only by `netlify/functions/translate.mjs`. Set it in Netlify **Site configuration → Environment variables** for deployed sites. For local development, copy [`netlify/.env.example`](netlify/.env.example) to a project-root `.env` and enter a valid key; `.env` is git-ignored. Set `OPENAI_MODEL` there or in Netlify if you want to override the default. Never put the OpenAI key in browser code or commit it to Git. Any previously exposed key should be revoked and replaced.

## Baybayin image OCR

The composer has a **Scan image** control. It sends a PNG, JPEG, WebP, or BMP (up to 4 MB) through the Netlify OCR proxy to the Python service in [`backend/ocr/`](backend/ocr/). The service uses the CNN weights and preprocessing code from the supplied OCR archive, returns recognized Baybayin characters, and the app sends that text through the existing translation flow. Images are processed in memory and are not stored by the OCR service. The character model can make mistakes, so the app shows its recognition confidence and asks users to review the result.

The OCR service must be deployed separately because the PyTorch runtime is too large for the current lightweight Netlify function. It includes a Dockerfile and can be deployed to Cloud Run or another container host. For Cloud Run, create a Secret Manager secret containing a long random token, then deploy from the repository root with `gcloud run deploy baybayin-ocr --source backend/ocr --region YOUR_REGION --memory 2Gi --cpu 2 --allow-unauthenticated --set-secrets=OCR_SERVICE_TOKEN=baybayin-ocr-token:latest`. The Cloud Run service must have permission to read that secret. Set the deployed HTTPS service URL as `OCR_SERVICE_URL` and the same token as `OCR_SERVICE_TOKEN` in Netlify's environment variables. For local development, copy `backend/ocr/.env.example` to `backend/ocr/.env` and `netlify/.env.example` to the repository-root `.env`; use the same token in both. From `backend/ocr`, run `pip install -r requirements.txt` and `uvicorn app:app --env-file .env --port 8000`, then run `npm run dev` from the repository root. The local Netlify proxy points to `http://localhost:8000` by default.

The attached archive reports 98.66% character accuracy and 67.33% exact-word accuracy on its evaluation set. Those results do not guarantee accuracy on new photos or handwriting; the CNN recognizes glyph shapes and does not replace the language translation model.

## Deploy
Import this folder into Netlify, keep the repository root as the base directory, and set Netlify's configuration file path to `netlify/netlify.toml`. The config publishes `.` and reads functions from `netlify/functions`. Set environment variables in Netlify and deploy. For local testing with Node.js 18+, run `npm install`, copy `netlify/.env.example` to a git-ignored root `.env`, add a valid key, and run `npm run dev`.

## About your OCR ZIP
The uploaded archive includes 37,000+ dataset entries, reference images, a Python/OpenCV preprocessing pipeline, PyTorch CNN and SVM artifacts. These models are Python artifacts and cannot run directly in static browser hosting. The current interface uses the character labels as a keyboard/reference and OpenAI for text translation; it does **not** claim to perform OCR with the uploaded CNN. To add image OCR, deploy the provided Python pipeline/model as a separate inference service and connect an upload control to that service.
