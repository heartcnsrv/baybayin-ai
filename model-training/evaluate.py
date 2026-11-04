"""Evaluation following the paper (Sections 3.2, 3.4 and 3.6).

    python evaluate.py split      # filter word images, 20% development / 80% test
    python evaluate.py tune       # tune segmentation thresholds on the development set
    python evaluate.py test       # final evaluation on the held-out test set
    python evaluate.py phrases    # supplementary phrase / sentence set

Word images ("Baybayin Word Images"): an image is kept only if it meets the
four inclusion criteria. Criterion 4 (the reference is an actual Tagalog word)
is checked against the Tagalog word list and Tagalog-English dictionaries.
Criteria 1-3 (standard Tagalog script, legible, horizontal and not heavily
stylised) need human judgement: researchers list failing images with a reason
in data/word_image_exclusions.csv. Every exclusion and its reason is recorded
in runs/word_image_filtering.csv.

The kept images are split with a fixed seed into a development set (20%,
used only for tuning) and a held-out test set (80%, used only in `test`).

Metrics (Section 3.6)
  character accuracy, CER   over Baybayin characters + kudlit marks
  WER, exact match          transliteration; a reading that differs only by
                            Baybayin's own ambiguity (e/i, o/u, d/r) is checked
                            against the intended Tagalog word, not counted as an error
  identification accuracy   chosen Tagalog word == reference; cases with a
                            correct transliteration but a wrong word are recorded
  English                   runs/english_grading_*.csv for researchers to mark
                            correct / partially correct / incorrect
  error stage               earliest stage where each wrong output went wrong
Each stage's output is saved per image in runs/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd

from baybayin_preprocess import Params
from translate_pipeline import (CLASS_CONSONANT, DATA_DIR, MODELS, GlyphObs, TParams, build_translator, syllable_options,
                                word_to_glyphs)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
WORD_DIR = os.path.join(DATA_DIR, "Baybayin Word Images")
SENTENCE_DIR = os.path.join(DATA_DIR, "Baybayin Sentence References Image")
REFERENCE_DIR = os.path.join(DATA_DIR, "Baybayin Reference Images")
SPLIT_FILE = os.path.join(HERE, "data", "word_image_split.csv")
EXCLUSIONS = os.path.join(HERE, "data", "word_image_exclusions.csv")
# Images the researchers reviewed and confirmed as actual Tagalog words although
# the automatic criterion-4 check found them in no reference list.
CONFIRMED = os.path.join(HERE, "data", "word_image_confirmed_words.csv")
PARAMS_FILE = os.path.join(MODELS, "pipeline_params.json")
SEED = 2024
DEV_SHARE = 0.20


# --------------------------------------------------------------------------- helpers

def label_from_filename(fn: str) -> str:
    """'1 Gwapo.PNG' -> 'gwapo'"""
    stem = os.path.splitext(os.path.basename(fn))[0]
    return re.sub(r"^\d+[\s_.\-]*", "", stem).strip().lower()


def edit_ops(ref, hyp):
    """Levenshtein distance and number of matched (correct) items in the best alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, correct = n, m, 0
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            correct += 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[n][m], correct


def reading_matches(obs, word: str) -> bool:
    """Is `word` one of the readings of these recognised characters?
    (e/i, o/u and d/r cannot be told apart in Baybayin.)"""
    pos = 0
    for g in obs:
        for opt in sorted(syllable_options(g), key=len, reverse=True):
            if word.startswith(opt, pos):
                pos += len(opt)
                break
        else:
            return False
    return pos == len(word)


def load_params() -> tuple[Params, TParams]:
    p, tp = Params(), TParams()
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            saved = json.load(f)
        for k, v in saved.get("preprocess", {}).items():
            setattr(p, k, v)
    return p, tp


# --------------------------------------------------------------------------- split

def cmd_split() -> None:
    from translate_pipeline import EnglishTranslator, Lexicon, tokenize
    # Criterion 4 evidence: the word list, the Tagalog-English dictionaries, or
    # the Tagalog text corpus (inflected forms like "nararamdaman" are missing
    # from the word list but are actual Tagalog words).
    corpus = pd.read_csv(os.path.join(DATA_DIR, "Tagalog English Dictionary Dataset", "Tagalog to English.csv"),
                         usecols=["tagalog"])["tagalog"].dropna().astype(str)
    counts = Counter(t for text in corpus for t in tokenize(text))
    words = (Lexicon.from_data_dir().words | set(EnglishTranslator(use_nmt=False).dict)
             | {w for w, c in counts.items() if c >= 2})
    manual = {}
    if os.path.exists(EXCLUSIONS):
        for r in csv.DictReader(open(EXCLUSIONS, encoding="utf-8")):
            if r.get("file"):
                manual[r["file"]] = r.get("reason", "excluded by researchers")
    confirmed = set()
    if os.path.exists(CONFIRMED):
        confirmed = {r["file"] for r in csv.DictReader(open(CONFIRMED, encoding="utf-8")) if r.get("file")}
    files = sorted(f for f in os.listdir(WORD_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    rows, kept = [], []
    for fn in files:
        ref = label_from_filename(fn)
        reason = manual.get(fn, "")
        if not reason:
            if not re.fullmatch(r"[a-zñ]+", ref):
                reason = "criterion 4: reference is not a single Tagalog word (e.g. hyphen, number)"
            elif ref not in words and fn not in confirmed:
                reason = "criterion 4: not found in the word list, dictionaries or Tagalog corpus (review)"
            elif word_to_glyphs(ref) is None:
                reason = "criterion 1: reference uses letters outside the standard Tagalog script"
        rows.append({"file": fn, "reference": ref, "kept": int(not reason), "reason": reason})
        if not reason:
            kept.append(fn)
    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_dev = round(DEV_SHARE * len(kept))
    split = {fn: ("development" if i < n_dev else "test") for i, fn in enumerate(kept)}
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "word_image_filtering.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "reference", "kept", "reason"])
        w.writeheader()
        w.writerows(rows)
    with open(SPLIT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "reference", "set"])
        for fn in sorted(split):
            w.writerow([fn, label_from_filename(fn), split[fn]])
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    print(f"{len(files)} images, {len(kept)} kept, {len(files) - len(kept)} excluded")
    for r, c in reasons.most_common():
        print(f"  {c:4d}  {r}")
    print(f"development {n_dev}, test {len(kept) - n_dev} (seed {SEED}) -> {SPLIT_FILE}")


def load_split(which: str) -> list[tuple[str, str]]:
    df = pd.read_csv(SPLIT_FILE, dtype=str, keep_default_na=False)
    return [(r.file, r.reference) for r in df[df.set == which].itertuples()]


# --------------------------------------------------------------------------- word evaluation

def evaluate_words(tr, items, params: Params, out_name: str | None = None, english: bool = False) -> dict:
    stats = Counter()
    char_err = char_ok = char_total = 0
    word_err = word_total = 0
    stages = Counter()
    rows = []
    for fn, ref in items:
        out = tr.translate_image(os.path.join(WORD_DIR, fn), params=params, do_english=english)
        lines = out["lines"]
        obs = [g for line in lines for w in [line["stage1_characters"]] for word in w for g in word]
        obs = [GlyphObs(**g) for g in obs]
        pred_words = [w for line in lines for w in line["tagalog"].split()]
        tagalog = " ".join(pred_words)
        ref_glyphs = word_to_glyphs(ref)
        hyp_glyphs = [(g.cls, g.mark if g.cls in CLASS_CONSONANT else "none") for g in obs]
        e, ok = edit_ops(ref_glyphs, hyp_glyphs)
        char_err += e
        char_ok += ok
        char_total += len(ref_glyphs)
        w_e, _ = edit_ops([ref], pred_words)
        word_err += w_e
        word_total += 1
        translit_ok = len(pred_words) == 1 and reading_matches(obs, ref)
        ident_ok = tagalog == ref
        stats["transliteration_exact"] += translit_ok
        stats["identification_correct"] += ident_ok
        stats["translit_ok_word_wrong"] += translit_ok and not ident_ok
        # earliest stage where the output went wrong
        if ident_ok:
            stage = "correct"
        elif not obs:
            stage = "unclear image (nothing segmented)"
        elif [c for c, _ in hyp_glyphs] != [c for c, _ in ref_glyphs]:
            stage = "character recognition"
        elif [m for _, m in hyp_glyphs] != [m for _, m in ref_glyphs]:
            stage = "kudlit / virama"
        elif not translit_ok:
            stage = "transliteration (word segmentation)"
        else:
            stage = "Tagalog word identification"
        stages[stage] += 1
        row = {"file": fn, "reference": ref, "characters": " ".join(f"{c}{'' if m == 'none' else '+' + m}" for c, m in hyp_glyphs),
               "reference_characters": " ".join(f"{c}{'' if m == 'none' else '+' + m}" for c, m in ref_glyphs),
               "transliteration": " ".join(line["stage2_transliteration"] for line in lines),
               "candidates": [c for line in lines for cs in line["stage2_candidates"] for c in cs],
               "tagalog": tagalog, "unverified": [u for line in lines for u in line["unverified_words"]],
               "error_stage": stage}
        if english:
            row["english"] = " | ".join(line.get("stage4_english", {}).get("english", "") for line in lines)
            row["translation_method"] = " | ".join(line.get("stage4_english", {}).get("method", "") for line in lines)
        rows.append(row)
    n = max(1, len(items))
    summary = {
        "images": len(items),
        "character_accuracy": round(100 * char_ok / max(1, char_total), 2),
        "character_error_rate": round(100 * char_err / max(1, char_total), 2),
        "word_error_rate": round(100 * word_err / max(1, word_total), 2),
        "transliteration_exact_match": round(100 * stats["transliteration_exact"] / n, 2),
        "tagalog_identification_accuracy": round(100 * stats["identification_correct"] / n, 2),
        "transliteration_correct_but_word_wrong": stats["translit_ok_word_wrong"],
        "errors_by_earliest_stage": dict(stages.most_common()),
    }
    if out_name:
        os.makedirs(RUNS, exist_ok=True)
        with open(os.path.join(RUNS, out_name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_image": rows}, f, indent=2, ensure_ascii=False)
        if english:
            with open(os.path.join(RUNS, f"english_grading_{out_name}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "tagalog_reference", "system_tagalog", "system_english", "method",
                            "english_reference", "grade (correct / partially correct / incorrect)"])
                for r in rows:
                    w.writerow([r["file"], r["reference"], r["tagalog"], r.get("english", ""),
                                r.get("translation_method", ""), "", ""])
    return summary


def cmd_tune() -> None:
    """Grid search of segmentation thresholds on the DEVELOPMENT set only."""
    dev = load_split("development")
    tr = build_translator(use_nmt=False)
    best = None
    for mark_ratio in (0.2, 0.3, 0.4, 0.5, 0.6):
        for word_gap in (0.4, 0.6, 0.8, 1.0, 1.5):
            p = Params(mark_ratio=mark_ratio, word_gap_ratio=word_gap)
            s = evaluate_words(tr, dev, p)
            key = (s["tagalog_identification_accuracy"], -s["character_error_rate"])
            print(f"mark_ratio={mark_ratio} word_gap_ratio={word_gap}: identification "
                  f"{s['tagalog_identification_accuracy']}%  CER {s['character_error_rate']}%", flush=True)
            if best is None or key > best[0]:
                best = (key, {"mark_ratio": mark_ratio, "word_gap_ratio": word_gap}, s)
    with open(PARAMS_FILE, "w") as f:
        json.dump({"preprocess": best[1], "tuned_on": "development set", "development_result": best[2]}, f, indent=2)
    print(f"best on development set: {best[1]} -> {PARAMS_FILE}")


def cmd_tune_script() -> None:
    """Tune the script-separation threshold (share of a word's characters that
    must be Baybayin) on data not used for testing: the development word images
    (all Baybayin: should be kept) and the validation Latin pages (should be rejected)."""
    from baybayin_preprocess import load_image, process_image
    from train_script_classifier import LATIN_PAGES, _split
    tr = build_translator(use_nmt=False)
    params, _ = load_params()
    def shares(path, single=False):
        res = process_image(load_image(path), params, single_word=single)
        out = []
        for line in res.lines:
            for w in line.words:
                if w.glyphs:
                    out.append(float(np.mean(tr.script.is_baybayin(np.stack([g.image for g in w.glyphs])))))
        return out
    dev = [max(shares(os.path.join(WORD_DIR, fn)) or [0.0]) for fn, _ in load_split("development")]
    latin = [s for fn in sorted(os.listdir(LATIN_PAGES))
             if fn.lower().endswith((".jpg", ".jpeg", ".png")) and _split("page:" + fn) == "val"
             for s in shares(os.path.join(LATIN_PAGES, fn))]
    best = None
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        kept = float(np.mean([s >= thr for s in dev]))
        rejected = float(np.mean([s < thr for s in latin]))
        score = (kept + rejected) / 2
        print(f"threshold {thr}: development Baybayin words kept {kept:.1%}, "
              f"validation Latin words rejected {rejected:.1%}")
        if best is None or score > best[0]:
            best = (score, thr, kept, rejected)
    saved = json.load(open(PARAMS_FILE)) if os.path.exists(PARAMS_FILE) else {}
    saved["script_min_share"] = best[1]
    saved["script_tuning"] = {"baybayin_kept": best[2], "latin_rejected": best[3],
                              "data": "development word images + validation Latin pages"}
    with open(PARAMS_FILE, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"best threshold {best[1]} -> {PARAMS_FILE}")


def cmd_test() -> None:
    test = load_split("test")
    params, _ = load_params()
    tr = build_translator(use_nmt=False)
    s = evaluate_words(tr, test, params, out_name="word_test", english=True)
    print(json.dumps(s, indent=2))


# --------------------------------------------------------------------------- supplementary phrases

def supplementary_items() -> list[tuple[str, str]]:
    items = []
    gt = os.path.join(HERE, "data", "sentence_transcriptions.csv")
    if os.path.exists(gt):
        for r in pd.read_csv(gt, dtype=str).itertuples():
            items.append((os.path.join(SENTENCE_DIR, f"{r.id}.jpg"), r.text))
    if os.path.isdir(REFERENCE_DIR):
        for fn in sorted(os.listdir(REFERENCE_DIR)):
            txt = os.path.join(REFERENCE_DIR, os.path.splitext(fn)[0] + ".txt")
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.exists(txt):
                items.append((os.path.join(REFERENCE_DIR, fn), open(txt, encoding="utf-8").read()))
    return [(p, " ".join(re.findall(r"[a-zñ'\-]+", t.lower()))) for p, t in items]


def cmd_phrases() -> None:
    params, _ = load_params()
    tr = build_translator(use_nmt=True)
    rows, err_w, tot_w, err_c, tot_c, exact = [], 0, 0, 0, 0, 0
    for path, ref in supplementary_items():
        out = tr.translate_image(path, params=params)
        hyp = " ".join(line["tagalog"] for line in out["lines"])
        ref_words, hyp_words = ref.replace("'", "").split(), hyp.split()
        we, _ = edit_ops(ref_words, hyp_words)
        ce, _ = edit_ops(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        err_w += we
        tot_w += len(ref_words)
        err_c += ce
        tot_c += len(ref.replace(" ", ""))
        exact += hyp_words == ref_words
        english = " | ".join(line.get("stage4_english", {}).get("english", "") for line in out["lines"])
        rows.append({"image": os.path.relpath(path, DATA_DIR), "tagalog_reference": ref, "system_tagalog": hyp,
                     "system_english": english, "wer": round(100 * we / max(1, len(ref_words)), 1)})
    summary = {"images": len(rows), "word_error_rate": round(100 * err_w / max(1, tot_w), 2),
               "transliteration_cer": round(100 * err_c / max(1, tot_c), 2),
               "exact_match": round(100 * exact / max(1, len(rows)), 2)}
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, "phrases.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RUNS, "english_grading_phrases.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "tagalog_reference", "system_tagalog", "system_english", "english_reference",
                    "grade (correct / partially correct / incorrect)"])
        for r in rows:
            w.writerow([r["image"], r["tagalog_reference"], r["system_tagalog"], r["system_english"], "", ""])
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "tune", "tune-script", "test", "phrases"])
    args = ap.parse_args()
    {"split": cmd_split, "tune": cmd_tune, "tune-script": cmd_tune_script, "test": cmd_test,
     "phrases": cmd_phrases}[args.cmd]()


if __name__ == "__main__":
    main()
