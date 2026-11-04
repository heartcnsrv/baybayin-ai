// Netlify build: copy only the website files into dist/. The datasets, models
// and training code are never published; the OCR function reads its model
// files from netlify/ocr-data (bundled via netlify.toml included_files).
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const files = [
  "index.html",
  "css/styles.css",
  "js/app.js",
  "database/firebase-config.js",
  "assets/logo.svg",
  "assets/favicon.svg",
];

rmSync(dist, { recursive: true, force: true });
for (const f of files) {
  mkdirSync(dirname(join(dist, f)), { recursive: true });
  cpSync(join(root, f), join(dist, f));
}

console.log(`dist/ ready (${files.length} files)`);
