"""Export the trained models and language data for the website's JavaScript
OCR function (netlify/functions/ocr.mjs, netlify/lib/baybai.mjs).

    python export_web_models.py        # -> ../netlify/ocr-data/

The models are NOT retrained or changed: the same weights are written in a
compact form that plain JavaScript can run.
  cnn.json / cnn.bin                base-character CNN (batch norm folded into the convolutions)
  diacritic.json / diacritic.bin    diacritic CNN (above / below / virama)
  script.json / script_bits.bin     Baybayin-vs-Latin SVM (RBF): scaler, coefficients, and the
                                    binary 28x28 support vectors packed as bits
  language.json                     Tagalog word list, corpus word / word-pair counts
  dictionary.json                   Tagalog -> English dictionary (single words)
  params.json                       thresholds tuned on the development set
Re-run after retraining any model.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch

from translate_pipeline import MODELS, EnglishTranslator, Lexicon, TParams, build_lm, load_params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "netlify", "ocr-data")


def export_cnn(pt: str, name: str) -> None:
    """SmallCNN: [conv-bn-relu x2, maxpool] x2, flatten, fc-relu, fc. BN is folded."""
    ck = torch.load(pt, map_location="cpu", weights_only=True)
    sd = {k: v.double() for k, v in ck["state_dict"].items() if not k.endswith("num_batches_tracked")}
    layers, blob, offset = [], [], 0

    def add(arr: np.ndarray) -> dict:
        nonlocal offset
        a = np.ascontiguousarray(arr, dtype=np.float32)
        blob.append(a.tobytes())
        ref = {"offset": offset, "shape": list(a.shape)}
        offset += a.nbytes
        return ref

    for block in (0, 1):
        for conv, bn in ((0, 1), (3, 4)):
            p, q = f"features.{block}.{conv}", f"features.{block}.{bn}"
            s = sd[f"{q}.weight"] / torch.sqrt(sd[f"{q}.running_var"] + 1e-5)
            w = sd[f"{p}.weight"] * s[:, None, None, None]
            b = (sd[f"{p}.bias"] - sd[f"{q}.running_mean"]) * s + sd[f"{q}.bias"]
            layers.append({"type": "conv3x3_relu", "weight": add(w.numpy()), "bias": add(b.numpy())})
        layers.append({"type": "maxpool2"})
    layers.append({"type": "dense_relu", "weight": add(sd["head.2.weight"].numpy()), "bias": add(sd["head.2.bias"].numpy())})
    layers.append({"type": "dense", "weight": add(sd["head.5.weight"].numpy()), "bias": add(sd["head.5.bias"].numpy())})
    with open(os.path.join(OUT, f"{name}.bin"), "wb") as f:
        f.write(b"".join(blob))
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"classes": list(ck["classes"]), "input": [1, 28, 28], "layers": layers}, f)


def export_svm() -> None:
    m = joblib.load(os.path.join(MODELS, "script_svm.joblib"))["model"]   # our own trained file
    scaler, svc = m[0], m[-1]
    raw = svc.support_vectors_ * scaler.scale_ + scaler.mean_            # back to 0/1 pixels
    bits = np.rint(raw).astype(np.uint8)
    assert np.abs(raw - bits).max() < 1e-3 and set(np.unique(bits)) <= {0, 1}, "support vectors are not binary"
    np.packbits(bits, axis=1).tofile(os.path.join(OUT, "script_bits.bin"))
    with open(os.path.join(OUT, "script.json"), "w") as f:
        json.dump({"n_support": int(bits.shape[0]), "n_features": int(bits.shape[1]),
                   "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "gamma": float(svc._gamma), "dual_coef": svc.dual_coef_[0].tolist(),
                   "intercept": float(svc.intercept_[0]), "classes": [int(c) for c in svc.classes_],
                   "positive_class": 1}, f)


def export_language() -> None:
    p = TParams()
    lex = Lexicon.from_data_dir()
    lm = build_lm(lex, p, cache=os.path.join(MODELS, "lm.pkl"))
    with open(os.path.join(OUT, "language.json"), "w", encoding="utf-8") as f:
        json.dump({"words": sorted(lex.words), "unigram": dict(lm.uni), "total": lm.total,
                   "k": lm.k, "lam": lm.lam, "vocab_size": len(lm.vocab),
                   "bigram": {f"{a}\t{b}": c for (a, b), c in lm.bi.items()},
                   "prev_count": dict(lm.prev_count)}, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(OUT, "dictionary.json"), "w", encoding="utf-8") as f:
        json.dump(EnglishTranslator(use_nmt=False).dict, f, ensure_ascii=False, separators=(",", ":"))
    pre = load_params()
    tuned = {}
    path = os.path.join(MODELS, "pipeline_params.json")
    if os.path.exists(path):
        tuned = json.load(open(path))
    with open(os.path.join(OUT, "params.json"), "w") as f:
        json.dump({"preprocess": {k: getattr(pre, k) for k in pre.__dataclass_fields__},
                   "script_min_share": tuned.get("script_min_share", p.script_min_share),
                   "max_candidates": p.max_candidates, "sentence_beam": p.sentence_beam}, f, indent=2)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    export_cnn(os.path.join(MODELS, "cnn.pt"), "cnn")
    export_cnn(os.path.join(MODELS, "diacritic_cnn.pt"), "diacritic")
    export_svm()
    export_language()
    for fn in sorted(os.listdir(OUT)):
        print(f"{os.path.getsize(os.path.join(OUT, fn)) / 1e6:8.2f} MB  {fn}")


if __name__ == "__main__":
    main()
