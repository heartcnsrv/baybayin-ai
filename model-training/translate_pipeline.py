"""
translate_pipeline.py
=====================
Baybayin -> Tagalog -> English after segmentation (paper Section 3.5), based on
the original project's module.

Stages (each stage's output is kept in the returned dict, so errors can be
traced to the stage where they occurred)
  1  character + diacritic   CNN (17 base classes) on each 28x28 base character;
                             diacritic classifier on each attached mark
                             (kudlit above / kudlit below / virama)
  2  candidate readings      every recognised character mapped to all of its
                             Latin readings (no mark -> Ca, above -> Ce/Ci,
                             below -> Co/Cu, virama -> C; da also reads ra);
                             combined per word
  3  Tagalog word            candidates not in the Tagalog word list are
                             discarded; one left -> chosen; several -> ranked
                             by frequency in a Tagalog text corpus, and for
                             multi-word input by fit with the neighbouring word
                             (bigram); none -> the most probable reading,
                             marked unverified
  4  English                 single word: Tagalog-English dictionary lookup;
                             phrase/sentence: pretrained neural machine
                             translation (NLLB-200)

Usage
-----
  python translate_pipeline.py image path.png [--word]
  python translate_pipeline.py selftest --n 2000        # oracle test (paper Table 4)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The translation model's packages and files live outside the C drive:
# D:/baybai-nmt/python-packages (transformers, ...) and D:/baybai-nmt/huggingface
# (the downloaded NLLB-200 model). Override with the BAYBAI_NMT_DIR variable.
NMT_DIR = os.environ.get("BAYBAI_NMT_DIR", "D:/baybai-nmt")
DATA_DIR = os.path.join(HERE, "..", "assets")
MODELS = os.path.join(HERE, "models")

# --------------------------------------------------------------------------- #
# Reading table (Section 3.5)
# --------------------------------------------------------------------------- #
CLASS_CONSONANT: Dict[str, List[str]] = {
    "ba": ["b"], "dara": ["d", "r"], "ga": ["g"], "ha": ["h"], "ka": ["k"],
    "la": ["l"], "ma": ["m"], "na": ["n"], "nga": ["ng"], "pa": ["p"],
    "sa": ["s"], "ta": ["t"], "wa": ["w"], "ya": ["y"],
}
VOWEL_CLASS: Dict[str, List[str]] = {"a": ["a"], "ei": ["e", "i"], "ou": ["o", "u"]}
MARK_VOWELS: Dict[str, List[str]] = {"none": ["a"], "above": ["e", "i"], "below": ["o", "u"], "virama": [""]}

# Baybayin Unicode for the recognised characters (output display only).
UNI_BASE = {"a": "ᜀ", "ei": "ᜁ", "ou": "ᜂ", "ka": "ᜃ", "ga": "ᜄ", "nga": "ᜅ",
            "ta": "ᜆ", "dara": "ᜇ", "na": "ᜈ", "pa": "ᜉ", "ba": "ᜊ", "ma": "ᜋ",
            "ya": "ᜌ", "la": "ᜎ", "wa": "ᜏ", "sa": "ᜐ", "ha": "ᜑ"}
UNI_MARK = {"none": "", "above": "ᜒ", "below": "ᜓ", "virama": "᜔"}


@dataclass
class TParams:
    lam: float = 0.7            # bigram / unigram interpolation for neighbouring words
    k: float = 0.1              # add-k smoothing for unigram frequencies
    max_candidates: int = 5000  # cap on readings per word (2^n grows fast for long words)
    sentence_beam: int = 8      # word sequences kept while ranking phrases
    n_best: int = 5
    script_min_share: float = 0.5  # a word is read only if this share of its characters is Baybayin


# --------------------------------------------------------------------------- #
# Resources: Tagalog word list, corpus frequencies, dictionary
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zñ]+", text.lower())


class Lexicon:
    """The Tagalog word list."""

    def __init__(self, words: Iterable[str]):
        self.words = {w for w in (str(x).strip().lower() for x in words)
                      if w and re.fullmatch(r"[a-zñ]+", w)}

    @classmethod
    def from_data_dir(cls, data_dir: str = DATA_DIR) -> "Lexicon":
        import pandas as pd
        path = os.path.join(data_dir, "Tagalog Dictionary", "tagalog words.csv")
        # This file has NO header row; the first word would be lost otherwise.
        return cls(pd.read_csv(path, header=None, usecols=[0])[0].dropna().astype(str))


class BigramLM:
    """Word frequencies and word-pair counts from a Tagalog text corpus.
    P(w | prev) = lam * c(prev, w) / c(prev) + (1 - lam) * P_uni(w)  (add-k unigram)."""

    def __init__(self, vocab: Iterable[str], k: float = 0.1, lam: float = 0.7):
        self.vocab = set(vocab)
        self.k, self.lam = k, lam
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.prev_count: Counter = Counter()
        self.total = 0

    def _map(self, toks: List[str]) -> List[str]:
        return [t if t in self.vocab else "<unk>" for t in toks]

    def fit_texts(self, texts: Iterable[str]) -> None:
        for text in texts:
            for sent in re.split(r"[.!?\n;:]+", str(text)):
                toks = self._map(tokenize(sent))
                if not toks:
                    continue
                seq = ["<s>"] + toks
                self.uni.update(toks)
                self.total += len(toks)
                for a, b in zip(seq, seq[1:]):
                    self.bi[(a, b)] += 1
                    self.prev_count[a] += 1

    def p_uni(self, w: str) -> float:
        V = len(self.vocab) + 1
        return (self.uni.get(w, 0) + self.k) / (self.total + self.k * V)

    def logp(self, w: str, prev: Optional[str]) -> float:
        """prev=None -> frequency (unigram) only."""
        pu = self.p_uni(w)
        if prev is None:
            return math.log(pu)
        cprev = self.prev_count.get(prev, 0)
        pb = self.bi.get((prev, w), 0) / cprev if cprev else 0.0
        lam = self.lam if cprev else 0.0
        return math.log(lam * pb + (1.0 - lam) * pu)

    def key(self, w: str) -> str:
        return w if w in self.vocab else "<unk>"


def build_lm(lex: Lexicon, p: TParams, data_dir: str = DATA_DIR, cache: Optional[str] = None) -> BigramLM:
    """Counts from the Tagalog side of 'Tagalog to English.csv' (Tagalog text corpus)."""
    import pandas as pd
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:   # our own cache file, written below
            lm = pickle.load(f)
        if lm.vocab == lex.words:
            lm.k, lm.lam = p.k, p.lam
            return lm
    lm = BigramLM(lex.words, p.k, p.lam)
    path = os.path.join(data_dir, "Tagalog English Dictionary Dataset", "Tagalog to English.csv")
    lm.fit_texts(pd.read_csv(path, usecols=["tagalog"])["tagalog"].dropna().astype(str))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(lm, f, protocol=4)
    return lm


class EnglishTranslator:
    """Single words: Tagalog-English dictionary lookup. Phrases and sentences:
    pretrained neural machine translation (NLLB-200), because a dictionary
    cannot handle word order and grammar."""

    def __init__(self, data_dir: str = DATA_DIR, use_nmt: bool = True,
                 nmt_name: str = "facebook/nllb-200-distilled-600M"):
        import pandas as pd
        self.dict: Dict[str, str] = {}
        base = os.path.join(data_dir, "Tagalog English Dictionary Dataset")
        p2 = os.path.join(base, "tagalog__english_dictionary.csv")
        if os.path.exists(p2):
            df = pd.read_csv(p2, encoding="utf-8-sig").dropna()
            for w, d in zip(df["tagalog_word"].astype(str).str.lower(), df["english_definition"].astype(str)):
                gloss = re.sub(r"^\S+\s+(?:(?:n|v|adj|adv|prep|conj|pron|interj|part)\.?,?\s*)+", "", d,
                               flags=re.IGNORECASE)
                self.dict.setdefault(w, gloss.strip())
        p1 = os.path.join(data_dir, "Tagalog Dictionary", "diksyo_tagalog_vocabulary.csv")
        if os.path.exists(p1):
            df = pd.read_csv(p1, usecols=["word", "translation"]).dropna()
            senses = defaultdict(list)
            for w, t in zip(df["word"].astype(str).str.lower(), df["translation"].astype(str)):
                if t not in senses[w]:
                    senses[w].append(t)
            self.dict.update({w: "; ".join(s) for w, s in senses.items()})
        self.use_nmt, self.nmt_name = use_nmt, nmt_name
        self._tok = self._model = None

    def _load_nmt(self) -> bool:
        if self._model is not None:
            return True
        import sys
        pkgs = os.path.join(NMT_DIR, "python-packages")
        if os.path.isdir(pkgs) and pkgs not in sys.path:
            sys.path.append(pkgs)   # appended: the main Python's numpy/torch stay in charge
        os.environ.setdefault("HF_HOME", os.path.join(NMT_DIR, "huggingface"))
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.nmt_name, src_lang="tgl_Latn")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.nmt_name)
            return True
        except Exception as e:  # transformers not installed / model not downloaded
            print(f"[translator] NMT unavailable ({type(e).__name__}); phrases use dictionary glosses")
            self.use_nmt = False
            return False

    def translate(self, words: List[str]) -> Dict[str, str]:
        if not words:
            return {"english": "", "method": "none"}
        if len(words) == 1:
            w = words[0]
            return ({"english": self.dict[w], "method": "dictionary"} if w in self.dict
                    else {"english": "", "method": "dictionary (not found)"})
        if self.use_nmt and self._load_nmt():
            inp = self._tok(" ".join(words), return_tensors="pt")
            out = self._model.generate(**inp, forced_bos_token_id=self._tok.convert_tokens_to_ids("eng_Latn"),
                                       max_new_tokens=128, num_beams=4)
            return {"english": self._tok.batch_decode(out, skip_special_tokens=True)[0], "method": "nmt (nllb-200)"}
        glosses = [self.dict.get(w, f"[{w}]").split(";")[0] for w in words]
        return {"english": " ".join(glosses), "method": "dictionary glosses (NMT unavailable)"}


# --------------------------------------------------------------------------- #
# Stage 1: recognised characters
# --------------------------------------------------------------------------- #
@dataclass
class GlyphObs:
    cls: str                    # CNN top-1 base class
    mark: str = "none"          # diacritic classifier label (or "none")
    prob: float = 1.0           # CNN probability of cls (reported, not used for ranking)


def baybayin_unicode(obs: Sequence[GlyphObs]) -> str:
    out = []
    for g in obs:
        out.append(UNI_BASE[g.cls] + (UNI_MARK[g.mark] if g.cls in CLASS_CONSONANT else ""))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Stage 2-3: candidate readings, word list, ranking
# --------------------------------------------------------------------------- #
def syllable_options(g: GlyphObs) -> List[str]:
    """All Latin readings of one recognised character with its diacritic."""
    if g.cls in VOWEL_CLASS:
        return list(VOWEL_CLASS[g.cls])
    return [c + v for c in CLASS_CONSONANT[g.cls] for v in MARK_VOWELS[g.mark]]


def most_probable_reading(obs: Sequence[GlyphObs]) -> str:
    """Reading kept when no candidate is in the word list: one reading per
    character, using the more frequent Tagalog letter of each ambiguous pair
    (i over e, u over o, d over r)."""
    out = []
    for g in obs:
        opts = syllable_options(g)
        preferred = [o for o in opts if not o.startswith("r") and not o.endswith(("e", "o"))]
        out.append((preferred or opts)[0])
    return "".join(out)


@dataclass
class Cand:
    word: str
    unverified: bool = False


def search_word(obs: Sequence[GlyphObs], lex: Lexicon, p: TParams) -> List[Cand]:
    """All combinations of the characters' readings; keep those in the word list.
    If none is in the list, keep the most probable reading, marked unverified."""
    if not obs:
        return []
    options = [syllable_options(g) for g in obs]
    found = []
    for combo in itertools.islice(itertools.product(*options), p.max_candidates):
        w = "".join(combo)
        if w in lex.words:
            found.append(w)
    if found:
        return [Cand(w) for w in dict.fromkeys(found)]
    return [Cand(most_probable_reading(obs), unverified=True)]


def rank_sequence(word_cands: List[List[Cand]], lm: BigramLM, p: TParams) -> List[Tuple[float, List[Cand]]]:
    """Rank by corpus frequency; for multi-word input also by how well each word
    fits the previous word (bigram). Beam search over the word sequence."""
    hyps: List[Tuple[float, List[Cand]]] = [(0.0, [])]
    for cands in word_cands:
        if not cands:
            continue
        nxt = []
        for sc, seq in hyps:
            prev = lm.key(seq[-1].word) if seq else None
            for c in cands:
                nxt.append((sc + lm.logp(lm.key(c.word), prev), seq + [c]))
        hyps = sorted(nxt, key=lambda t: -t[0])[:p.sentence_beam]
    return hyps


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
class ScriptClassifier:
    """Baybayin vs. not-Baybayin SVM on the 28x28 characters (models/script_svm.joblib),
    the script-separation step of Pino et al. [1, 3]."""

    def __init__(self, path: str):
        import joblib
        self.model = joblib.load(path)["model"]   # our own trained file

    def is_baybayin(self, images_uint8: np.ndarray) -> np.ndarray:
        X = np.asarray(images_uint8, np.float32).reshape(len(images_uint8), -1) / 255.0
        return self.model.predict(X) == 1


class CNNClassifier:
    """Base-character CNN (models/cnn.pt) or diacritic CNN (models/diacritic_cnn.pt)."""

    def __init__(self, path: str):
        from train_models import load_cnn
        self.model, self.classes = load_cnn(path)

    def predict(self, images_uint8: np.ndarray) -> Tuple[List[str], np.ndarray]:
        from train_models import predict_glyphs
        return predict_glyphs(self.model, self.classes, images_uint8)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class BaybayinTranslator:
    def __init__(self, lex: Lexicon, lm: BigramLM, params: Optional[TParams] = None,
                 translator: Optional[EnglishTranslator] = None,
                 classifier: Optional[CNNClassifier] = None, diacritics: Optional[CNNClassifier] = None,
                 script: Optional[ScriptClassifier] = None):
        self.lex, self.lm = lex, lm
        self.p = params or TParams()
        self.translator, self.clf, self.diacritics, self.script = translator, classifier, diacritics, script

    def observations_from_result(self, res):
        """Segmentation result -> (lines of words of recognised characters,
        the kept segmentation words per line, the words rejected as not Baybayin)."""
        lines, kept_words, rejected = [], [], []
        for line in res.lines:
            words_obs, line_words = [], []
            for w in line.words:
                if not w.glyphs:
                    continue
                if self.script is not None:
                    share = float(np.mean(self.script.is_baybayin(np.stack([g.image for g in w.glyphs]))))
                    if share < self.p.script_min_share:
                        rejected.append(w)
                        continue
                line_words.append(w)
                names, probs = self.clf.predict(np.stack([g.image for g in w.glyphs]))
                marks = ["none"] * len(w.glyphs)
                marked = [i for i, g in enumerate(w.glyphs) if g.marks]
                if marked and self.diacritics is not None:
                    # several marks on one character: the largest one is classified
                    imgs = [max(w.glyphs[i].marks, key=lambda m: m.bbox[2] * m.bbox[3]).image for i in marked]
                    labels, _ = self.diacritics.predict(np.stack(imgs))
                    for i, lab in zip(marked, labels):
                        marks[i] = lab
                words_obs.append([GlyphObs(names[i], marks[i] if names[i] in CLASS_CONSONANT else "none",
                                           float(probs[i].max())) for i in range(len(w.glyphs))])
            if words_obs:
                lines.append(words_obs)
                kept_words.append(line_words)
        return lines, kept_words, rejected

    def translate_observations(self, words_obs: List[List[GlyphObs]], do_english: bool = True) -> Dict:
        t0 = time.time()
        word_cands = [search_word(w, self.lex, self.p) for w in words_obs]
        ranked = rank_sequence(word_cands, self.lm, self.p)
        _, best = ranked[0] if ranked else (0.0, [])
        words = [c.word for c in best]
        log = {
            "stage1_characters": [[asdict(g) for g in w] for w in words_obs],
            "baybayin": " ".join(baybayin_unicode(w) for w in words_obs),
            "stage2_transliteration": " ".join(most_probable_reading(w) for w in words_obs),
            "stage2_candidates": [[c.word for c in cs[:10]] for cs in word_cands],
            "stage3_ranked": [{"score": round(s, 3), "words": [c.word for c in seq]}
                              for s, seq in ranked[:self.p.n_best]],
            "tagalog": " ".join(words),
            "unverified_words": [c.word for c in best if c.unverified],
        }
        if do_english and self.translator and words:
            log["stage4_english"] = self.translator.translate(words)
        log["seconds"] = round(time.time() - t0, 3)
        return log

    def translate_image(self, image, single_word: bool = False, params=None, do_english: bool = True) -> Dict:
        from baybayin_preprocess import load_image, process_image
        if isinstance(image, str):
            image = load_image(image)
        res = process_image(image, p=params, single_word=single_word)
        obs_lines, kept_words, rejected = self.observations_from_result(res)
        if self.script is not None:
            # Text can be darker or lighter than its background (e.g. white
            # lettering on a photo). Also read the other side of the Otsu
            # threshold and keep whichever yields more Baybayin characters.
            for light in (True, False):
                alt = process_image(image, p=params, single_word=single_word, light_text=light)
                alt_obs = self.observations_from_result(alt)
                if _n_chars(alt_obs[0]) > _n_chars(obs_lines):
                    res, (obs_lines, kept_words, rejected) = alt, alt_obs
        lines = []
        for wo, words in zip(obs_lines, kept_words):
            log = self.translate_observations(wo, do_english)
            log["word_boxes"] = [list(w.bbox) for w in words]
            lines.append(log)
        return {"lines": lines, "segmentation": res, "rejected_boxes": [list(w.bbox) for w in rejected]}


def _n_chars(obs_lines) -> int:
    return sum(len(w) for line in obs_lines for w in line)


def build_translator(use_nmt: bool = True, params: Optional[TParams] = None) -> BaybayinTranslator:
    p = params or TParams()
    tuned = os.path.join(MODELS, "pipeline_params.json")
    if os.path.exists(tuned):
        with open(tuned) as f:
            p.script_min_share = json.load(f).get("script_min_share", p.script_min_share)
    lex = Lexicon.from_data_dir()
    lm = build_lm(lex, p, cache=os.path.join(MODELS, "lm.pkl"))
    diacritic_path = os.path.join(MODELS, "diacritic_cnn.pt")
    script_path = os.path.join(MODELS, "script_svm.joblib")
    return BaybayinTranslator(lex, lm, p, EnglishTranslator(use_nmt=use_nmt),
                              CNNClassifier(os.path.join(MODELS, "cnn.pt")),
                              CNNClassifier(diacritic_path) if os.path.exists(diacritic_path) else None,
                              ScriptClassifier(script_path) if os.path.exists(script_path) else None)


# --------------------------------------------------------------------------- #
# Oracle self-test (paper Table 4): perfect character observations
# --------------------------------------------------------------------------- #
def word_to_glyphs(word: str) -> Optional[List[Tuple[str, str]]]:
    """Baybayin spelling of a Tagalog word as (class, mark) per character, with
    a virama for consonants without a vowel. None if a letter has no Baybayin form."""
    vow = "aeiou"
    mark_of = {"a": "none", "e": "above", "i": "above", "o": "below", "u": "below"}
    cons_class = {"b": "ba", "d": "dara", "r": "dara", "g": "ga", "h": "ha", "k": "ka", "l": "la", "m": "ma",
                  "n": "na", "ng": "nga", "p": "pa", "s": "sa", "t": "ta", "w": "wa", "y": "ya"}
    out: List[Tuple[str, str]] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in vow:
            out.append(({"a": "a", "e": "ei", "i": "ei", "o": "ou", "u": "ou"}[ch], "none"))
            i += 1
            continue
        c, step = ("ng", 2) if word.startswith("ng", i) else (ch, 1)
        if c not in cons_class:
            return None
        if i + step < n and word[i + step] in vow:
            out.append((cons_class[c], mark_of[word[i + step]]))
            i += step + 1
        else:
            out.append((cons_class[c], "virama"))
            i += step
    return out


def selftest(n: int = 2000, seed: int = 0, p: Optional[TParams] = None) -> Dict:
    p = p or TParams()
    lex = Lexicon.from_data_dir()
    t0 = time.time()
    lm = build_lm(lex, p, cache=os.path.join(MODELS, "lm.pkl"))
    print(f"word list: {len(lex.words)} words; corpus: {lm.total} tokens, {len(lm.bi)} bigrams "
          f"({time.time() - t0:.0f}s)")
    pool = sorted(lex.words)
    random.Random(seed).shuffle(pool)
    tests = [(w, g) for w in pool for g in [word_to_glyphs(w)] if g][:n]
    top1 = top5 = recall = 0
    ncand, misses = [], []
    t0 = time.time()
    for w, glyphs in tests:
        cands = search_word([GlyphObs(c, m) for c, m in glyphs], lex, p)
        order = [c.word for c in sorted(cands, key=lambda c: -lm.logp(lm.key(c.word), None))]
        ncand.append(len(cands))
        recall += w in order
        top1 += bool(order) and order[0] == w
        top5 += w in order[:5]
        if order[:1] != [w] and len(misses) < 8:
            misses.append({"reference": w, "predicted": order[:3]})
    N = len(tests)
    res = {"words_tested": N, "candidate_recall": recall / N, "top1": top1 / N, "top5": top5 / N,
           "mean_candidates": float(np.mean(ncand)), "median_candidates": float(np.median(ncand)),
           "seconds": round(time.time() - t0, 1), "word_list_words": len(lex.words),
           "corpus_tokens": lm.total, "bigrams": len(lm.bi), "sample_misses": misses}
    print(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("image")
    s.add_argument("path")
    s.add_argument("--word", action="store_true", help="the image holds a single word")
    s.add_argument("--no-nmt", action="store_true")
    s = sub.add_parser("selftest")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=os.path.join(MODELS, "selftest.json"))
    args = ap.parse_args()

    if args.cmd == "selftest":
        res = selftest(args.n, args.seed)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        return
    tr = build_translator(use_nmt=not args.no_nmt)
    out = tr.translate_image(args.path, single_word=args.word)
    for i, line in enumerate(out["lines"], 1):
        print(f"Line {i}")
        print("  Baybayin:       " + line["baybayin"])
        print("  Transliteration:" + " " + line["stage2_transliteration"])
        print("  Tagalog:        " + line["tagalog"])
        if line.get("stage4_english"):
            print(f"  English:        {line['stage4_english']['english']}  ({line['stage4_english']['method']})")
        if line["unverified_words"]:
            print("  Unverified:     " + ", ".join(line["unverified_words"]))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Web / app entry point
# --------------------------------------------------------------------------- #
PARAMS_FILE = os.path.join(MODELS, "pipeline_params.json")


def load_params():
    """Preprocessing thresholds tuned on the development set (evaluate.py tune)."""
    from baybayin_preprocess import Params
    p = Params()
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            for k, v in json.load(f).get("preprocess", {}).items():
                setattr(p, k, v)
    return p


def recognize(tr: BaybayinTranslator, img: np.ndarray, params=None, do_english: bool = True) -> Dict:
    """Image -> recognised Baybayin, transliteration, Tagalog, English, word boxes
    (in the uploaded image's own coordinates) and a preview: the user's own
    photo with one green box around each word that was read as Baybayin."""
    import cv2
    out = tr.translate_image(img, params=params or load_params(), do_english=do_english)
    res = out["segmentation"]
    words, baybayin, tagalog, english, translit = [], [], [], [], []
    for li, line in enumerate(out["lines"]):
        chosen = line["tagalog"].split()
        for wi, (obs, box) in enumerate(zip(line["stage1_characters"], line["word_boxes"])):
            g = [GlyphObs(**o) for o in obs]
            x, y, w, h = res.to_source_box(tuple(box))
            word = chosen[wi] if wi < len(chosen) else ""
            words.append({"line": li, "baybayin": baybayin_unicode(g), "reading": most_probable_reading(g),
                          "tagalog": word, "verified": word not in line["unverified_words"],
                          "confidence": float(min(o.prob for o in g)), "box": [x, y, x + w, y + h]})
        baybayin.append(line["baybayin"])
        translit.append(line["stage2_transliteration"])
        tagalog.append(line["tagalog"])
        english.append(line.get("stage4_english", {}).get("english", ""))
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thick = max(2, round(max(overlay.shape[:2]) / 400))
    pad = 2 * thick
    for wd in words:
        x0, y0, x1, y1 = wd["box"]
        cv2.rectangle(overlay, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (60, 180, 60), thick, cv2.LINE_AA)
    ignored = [res.to_source_box(tuple(b)) for b in out["rejected_boxes"]]
    return {"baybayin": "\n".join(baybayin), "transliteration": "\n".join(translit),
            "tagalog": "\n".join(tagalog), "english": "\n".join(english), "words": words,
            "unverified": [w["tagalog"] for w in words if not w["verified"]],
            "ignored": [{"box": [x, y, x + w, y + h], "reason": "not Baybayin"} for x, y, w, h in ignored],
            "overlay": overlay}


def english_for(tr: BaybayinTranslator, tagalog: str) -> str:
    """English for an already identified Tagalog text, line by line (stage 4)."""
    return "\n".join(tr.translator.translate(line.split())["english"] if line.split() else ""
                     for line in tagalog.split("\n"))
