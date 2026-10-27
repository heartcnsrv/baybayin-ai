# Baybayin AI — Netlify web app

Plain HTML, CSS and JavaScript recreation of the supplied UI. Firebase is initialized using the project configuration supplied in chat; translation history is stored per anonymous Firebase user in Firestore. AI requests go through a Netlify Function so the OpenAI secret is not exposed to browser JavaScript.

## Structure
- `index.html` — app structure
- `css/styles.css` — responsive styling
- `js/app.js` — keyboard, translation UI, Firebase auth/database
- `assets/` — logo and favicon
- `netlify/functions/translate.mjs` — server-side OpenAI proxy
- `database/firestore.rules` — recommended per-user Firestore rules
- `reference-data/model-metadata/` — model class list and findings from supplied OCR ZIP

## Firebase setup
The provided Firebase config is already wired into `js/app.js`. In Firebase Console for `baybayin-ai`, enable **Authentication → Sign-in method → Anonymous and Google**, create/enable Firestore, and publish [`database/firestore.rules`](database/firestore.rules). Add your Netlify domain under Authentication → Settings → Authorized domains.

Translations live at `users/{uid}/translations/{documentId}` with source, Tagalog transliteration, English translation, and server timestamp. Firebase web config is not a server secret; access control comes from auth and Firestore rules.

## OpenAI key security
The API key pasted into chat must be considered exposed: revoke it and create a new key. In Netlify **Site configuration → Environment variables**, add `OPENAI_API_KEY` with the replacement key. Optionally set `OPENAI_MODEL` to `gpt-4o-mini`. Redeploy. Never place the OpenAI key in client-side JS or commit it to Git.

## Deploy
Import this folder into Netlify. Publish directory is `.` and functions directory is set in `netlify.toml`. Set environment variables and deploy. For local testing with Node.js 18+, run `npm install`, create a local `.env` from `.env.example` and run `npx netlify dev`.

## About your OCR ZIP
The uploaded archive includes 37,000+ dataset entries, reference images, a Python/OpenCV preprocessing pipeline, PyTorch CNN and SVM artifacts. These models are Python artifacts and cannot run directly in static browser hosting. The current interface uses the character labels as a keyboard/reference and OpenAI for text translation; it does **not** claim to perform OCR with the uploaded CNN. To add image OCR, deploy the provided Python pipeline/model as a separate inference service and connect an upload control to that service.
