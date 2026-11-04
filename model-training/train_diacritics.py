"""Diacritic classifier (paper Section 3.5): labels each mark attached to a
base character as a kudlit above, a kudlit below, or a cross-shaped virama.

    python train_diacritics.py

Data: "Baybayin Handwritten Images" (isolated characters with diacritics),
file names "<class>.<id>-<writer>.jpg", e.g. "ke_ki" = consonant + kudlit above,
"ko_ku" = kudlit below, "k" = virama. Each image goes through the same
preprocessing and segmentation as the full system; the attached mark's 28x28
context image (mark with its base character) is the classifier input.
Images where segmentation finds no separate mark (e.g. a mark touching its
character) are skipped and counted.

Model: the same CNN architecture, augmentation (rotation, scaling,
translation, stroke thickness), cross-entropy loss and early stopping on
validation accuracy as the base-character CNN (train_models.py).
Whole writers are held out for validation and testing.
Writes models/diacritic_cnn.pt, models/diacritic_metrics.json,
models/confusion_diacritic.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from baybayin_preprocess import load_image, process_image
from train_models import _predict, _to_tensor, _torch, augment, build_cnn, save_confusion, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "assets", "Baybayin Handwritten Images")
DIACRITICS = ["above", "below", "virama"]
NAME = re.compile(r"^(?P<cls>[a-z_]+)\.(?P<id>[^-]+)-(?P<writer>.+)$")
VOWELS = {"a", "e_i", "o_u"}
# Writers who drew most of the images always train; other writers are split
# by writer so validation/test measure unseen handwriting.
TRAIN_WRITERS = {"james", "neil"}
UPSCALE = 4


def mark_label(cls: str) -> str | None:
    """'ke_ki' -> above, 'ko_ku' -> below, 'k' -> virama, plain/vowel -> None."""
    if cls in VOWELS:
        return None
    if "_" in cls:
        first = cls.split("_")[0]
        return "above" if first.endswith(("e", "i")) else "below"
    return None if cls.endswith("a") else "virama"


def writer_split(writer: str) -> str:
    w = re.sub(r"[^a-z]", "", writer.lower())
    if w in TRAIN_WRITERS:
        return "train"
    r = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if r < 0.34 else "val" if r < 0.67 else "test"


def build_dataset():
    X, y, split, skipped = [], [], [], Counter()
    for fn in sorted(os.listdir(SRC)):
        if not fn.lower().endswith(".jpg"):
            continue  # also skips "*.jpg~HEAD" leftovers
        m = NAME.match(os.path.splitext(fn)[0])
        if not m:
            skipped["bad name"] += 1
            continue
        label = mark_label(m.group("cls"))
        if label is None:
            continue
        img = load_image(os.path.join(SRC, fn))
        # The source images are only 28x28, where a kudlit dot is a few pixels
        # and would be filtered as noise; resize 4x before preprocessing.
        img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        res = process_image(img, single_word=True)
        glyphs = [g for line in res.lines for w in line.words for g in w.glyphs]
        marked = [g for g in glyphs if g.marks]
        if len(marked) != 1:
            skipped["no separate mark found" if not marked else "several marked bases"] += 1
            continue
        mark = max(marked[0].marks, key=lambda mk: mk.bbox[2] * mk.bbox[3])
        X.append(mark.image)
        y.append(DIACRITICS.index(label))
        split.append(writer_split(m.group("writer")))
    return np.stack(X), np.array(y), np.array(split), skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, split, skipped = build_dataset()
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print(f"diacritic samples: train {len(tr)} val {len(va)} test {len(te)}; skipped {dict(skipped)}")
    print("per class:", {DIACRITICS[k]: int(v) for k, v in sorted(Counter(y).items())})

    torch, nn, _ = _torch()
    torch.manual_seed(args.seed)
    model = build_cnn(len(DIACRITICS))
    counts = np.bincount(y[tr], minlength=len(DIACRITICS)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(DIACRITICS) * np.maximum(counts, 1))))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr, ytr = _to_tensor(X[tr], "cpu"), torch.from_numpy(y[tr])
    best_acc, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(augment(Xtr[b])), ytr[b])
            loss.backward()
            opt.step()
        val_acc = float(accuracy_score(y[va], _predict(model, X[va], "cpu").argmax(1)))
        history.append({"epoch": ep, "val_acc": val_acc})
        print(f"  [diacritic] epoch {ep:>2}  val acc={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    pred = _predict(model, X[te], "cpu").argmax(1)
    res = summarize(y[te], pred, DIACRITICS)
    res.update({"best_val_acc": best_acc, "epochs_run": len(history),
                "samples": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                "skipped": dict(skipped)})
    torch.save({"state_dict": best_state, "classes": DIACRITICS}, os.path.join(args.out, "diacritic_cnn.pt"))
    with open(os.path.join(args.out, "diacritic_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_confusion(confusion_matrix(y[te], pred, labels=range(len(DIACRITICS))), DIACRITICS,
                   f"Diacritic CNN (test acc {res['accuracy']:.3f})", os.path.join(args.out, "confusion_diacritic.png"))
    print(f"[diacritic] TEST accuracy={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
