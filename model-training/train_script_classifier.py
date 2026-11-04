"""Baybayin vs. non-Baybayin character classifier (SVM), following the
script-separation step of Pino et al. [1, 3], who used an SVM to tell Baybayin
characters from Latin characters before recognition.

    python train_script_classifier.py

Input: the same 28x28 binary character images the CNN receives (output of
baybayin_preprocess). Classes: 0 = not Baybayin, 1 = Baybayin.
  Baybayin      the paper's character dataset (processed-data/paper_split),
                same train / val / test split as the CNN and SVM, plus the base
                characters cut from "Baybayin Handwritten Images" (more writers
                and pen styles), split by writer
  not Baybayin  every character-sized component that the paper's preprocessing
                cuts out of the "Tagalog Word Images" pages (handwritten and
                printed Latin letters, plus page junk such as lines and smudges),
                and Latin letters / digits rendered in Windows fonts.
                Split by page and by font, so test pages/fonts are never trained on.
SVM: RBF kernel on the flattened 28x28 pixels (as the paper's SVM baseline),
C chosen on the validation split.
Writes models/script_svm.joblib and models/script_metrics.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baybayin_preprocess import GLYPH_SIZE, Params, load_image, normalize_glyph, process_image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
LATIN_PAGES = os.path.join(ASSETS, "Tagalog Word Images")
PAPER_SPLIT = os.path.join(HERE, "..", "processed-data", "paper_split")
MODELS = os.path.join(HERE, "models")
FONT_DIR = "C:/Windows/Fonts"
FONTS = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "calibri.ttf", "calibrii.ttf", "Candara.ttf", "comic.ttf",
         "comicbd.ttf", "comici.ttf", "consola.ttf", "constan.ttf", "corbel.ttf", "Gabriola.ttf", "georgia.ttf",
         "georgiai.ttf", "Inkfree.ttf", "segoepr.ttf", "segoeprb.ttf", "segoesc.ttf", "segoescb.ttf",
         "segoeui.ttf", "tahoma.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf", "trebuc.ttf", "verdana.ttf"]
LATIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SEED = 42


def _split(key: str) -> str:
    r = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if r < 0.7 else "val" if r < 0.85 else "test"


def latin_page_glyphs():
    """Base-character components from the Latin-only pages, cut by the paper's
    preprocessing (so they look exactly like what the CNN would receive)."""
    X, split = [], []
    for fn in sorted(os.listdir(LATIN_PAGES)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            continue
        res = process_image(load_image(os.path.join(LATIN_PAGES, fn)), Params())
        s = _split("page:" + fn)
        for line in res.lines:
            for w in line.words:
                for g in w.glyphs:
                    X.append(g.image)
                    split.append(s)
    return X, split


def handwritten_baybayin_glyphs():
    """Base characters from "Baybayin Handwritten Images", cut by the paper's
    preprocessing (marks removed). Writer-disjoint split, as train_diacritics.py."""
    import re
    from train_diacritics import NAME, UPSCALE, writer_split
    src = os.path.join(ASSETS, "Baybayin Handwritten Images")
    X, split = [], []
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith(".jpg"):
            continue
        m = NAME.match(os.path.splitext(fn)[0])
        if not m:
            continue
        img = cv2.resize(load_image(os.path.join(src, fn)), None, fx=UPSCALE, fy=UPSCALE,
                         interpolation=cv2.INTER_CUBIC)
        res = process_image(img, Params(), single_word=True)
        glyphs = [g for line in res.lines for w in line.words for g in w.glyphs]
        if len(glyphs) == 1:   # one base character per image, as expected
            X.append(glyphs[0].image)
            split.append(writer_split(m.group("writer")))
    return X, split


def font_glyphs(per_font: int = 2):
    """Latin letters and digits rendered in many fonts, normalised like the CNN input."""
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(SEED)
    X, split = [], []
    for fname in FONTS:
        path = os.path.join(FONT_DIR, fname)
        if not os.path.exists(path):
            continue
        s = _split("font:" + fname)
        for ch in LATIN_CHARS:
            for _ in range(per_font):
                font = ImageFont.truetype(path, rng.randint(28, 64))
                l, t, r, b = font.getbbox(ch)
                im = Image.new("L", (r - l + 20, b - t + 20), 0)
                ImageDraw.Draw(im).text((10 - l, 10 - t), ch, font=font, fill=255)
                arr = np.array(im)
                h, w = arr.shape
                m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), 1.0)
                arr = cv2.warpAffine(arr, m, (w, h))
                X.append(normalize_glyph(arr > 127, GLYPH_SIZE))
                split.append(s)
    return X, split


def main() -> None:
    t0 = time.time()
    # Pickle-enabled load of our own dataset file (the paper's character data).
    d = np.load(os.path.join(PAPER_SPLIT, "symbols_dataset.npz"), allow_pickle=True)
    sp = np.load(os.path.join(PAPER_SPLIT, "split.npz"))
    hx, hs = handwritten_baybayin_glyphs()
    hx, hs = np.array(hx, np.uint8), np.array(hs)
    bay = {k: np.concatenate([d["X"][sp[k]], hx[hs == k]]) for k in ("train", "val", "test")}
    print(f"Baybayin: {len(d['X'])} symbol-dataset + {len(hx)} handwritten base characters", flush=True)

    lx, ls = latin_page_glyphs()
    fx, fs = font_glyphs()
    neg_X = np.array(lx + fx, np.uint8)
    neg_s = np.array(ls + fs)
    print(f"not-Baybayin: {len(lx)} components from Latin pages, {len(fx)} font characters "
          f"({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.RandomState(SEED)
    data = {}
    for k in ("train", "val", "test"):
        neg = neg_X[neg_s == k]
        pos = bay[k]
        # Balance the classes in training (Baybayin images far outnumber Latin ones).
        if k == "train" and len(pos) > 2 * len(neg):
            pos = pos[rng.choice(len(pos), 2 * len(neg), replace=False)]
        X = np.concatenate([pos, neg]).reshape(-1, GLYPH_SIZE * GLYPH_SIZE).astype(np.float32) / 255.0
        y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        data[k] = (X, y)
        print(f"  {k}: {len(pos)} Baybayin, {len(neg)} not Baybayin", flush=True)

    best = None
    for C in (1, 3, 10, 30):
        m = make_pipeline(StandardScaler(), SVC(C=C, class_weight="balanced"))
        m.fit(*data["train"])
        acc = accuracy_score(data["val"][1], m.predict(data["val"][0]))
        print(f"  [script svm] C={C:<3} val acc={acc:.4f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    model = best[2]
    Xt, yt = data["test"]
    pred = model.predict(Xt)
    res = {
        "C": best[1],
        "test_accuracy": round(float(accuracy_score(yt, pred)), 4),
        "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "baybayin_kept (recall)": round(float(recall_score(yt, pred, pos_label=1)), 4),
        "non_baybayin_rejected (recall)": round(float(recall_score(yt, pred, pos_label=0)), 4),
        "confusion [[non->non, non->bay], [bay->non, bay->bay]]": confusion_matrix(yt, pred).tolist(),
        "samples": {k: int(len(v[1])) for k, v in data.items()},
    }
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({"model": model, "classes": ["not_baybayin", "baybayin"]}, os.path.join(MODELS, "script_svm.joblib"))
    with open(os.path.join(MODELS, "script_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
