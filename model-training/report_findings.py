"""Write the findings tables (Markdown) from the saved evaluation results.

    python report_findings.py            # -> runs/findings.md

Reads models/metrics.json (CNN + SVM), models/diacritic_metrics.json,
models/script_metrics.json, models/selftest.json, models/pipeline_params.json,
runs/word_test.json and runs/phrases.json.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
RUNS = os.path.join(HERE, "runs")


def read(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(v: float) -> str:
    return f"{100 * v:.2f}%" if v <= 1 else f"{v:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "findings.md"))
    args = ap.parse_args()
    out = ["# Findings", ""]

    m = read(os.path.join(MODELS, "metrics.json"))
    if m:
        out += ["## Table 1. CNN and SVM character classification (paper test split)", "",
                "| Metric | CNN | SVM |", "|---|---|---|",
                f"| Classification accuracy | {pct(m['cnn']['accuracy'])} | {pct(m['svm']['accuracy'])} |",
                f"| Macro-averaged F1-score | {m['cnn']['macro_f1']:.4f} | {m['svm']['macro_f1']:.4f} |", ""]

    d = read(os.path.join(MODELS, "diacritic_metrics.json"))
    s = read(os.path.join(MODELS, "script_metrics.json"))
    if d or s:
        out += ["## Supporting classifiers", "", "| Classifier | Test accuracy | Macro-F1 | Note |", "|---|---|---|---|"]
        if d:
            out.append(f"| Diacritic CNN (above / below / virama) | {pct(d['accuracy'])} | {d['macro_f1']:.4f} | "
                       f"{d['samples']['test']} marks, unseen writers |")
        if s:
            out.append(f"| Script SVM (Baybayin vs. not) | {pct(s['test_accuracy'])} | {s['test_macro_f1']:.4f} | "
                       f"Baybayin kept {pct(s['baybayin_kept (recall)'])}, "
                       f"non-Baybayin rejected {pct(s['non_baybayin_rejected (recall)'])} |")
        out.append("")

    w = read(os.path.join(RUNS, "word_test.json"))
    if w:
        ws = w["summary"]
        out += ["## Table 2. Word images, held-out test set", "", "| Metric | Result |", "|---|---|",
                f"| Images tested | {ws['images']} |",
                f"| Character accuracy | {ws['character_accuracy']}% |",
                f"| Character error rate (CER) | {ws['character_error_rate']}% |",
                f"| Word error rate (WER) | {ws['word_error_rate']}% |",
                f"| Transliteration exact match | {ws['transliteration_exact_match']}% |",
                f"| Tagalog identification accuracy | {ws['tagalog_identification_accuracy']}% |",
                f"| Transliteration correct, word wrong | {ws['transliteration_correct_but_word_wrong']} |", "",
                "Errors by earliest stage:", "", "| Stage | Images |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in ws["errors_by_earliest_stage"].items()]
        out.append("")

    t = read(os.path.join(MODELS, "selftest.json"))
    if t:
        out += ["## Table 4. Candidate generation, oracle self-test", "", "| Metric | Result |", "|---|---|",
                f"| Words tested | {t['words_tested']} |",
                f"| Candidate recall | {pct(t['candidate_recall'])} |",
                f"| Top-1 accuracy | {pct(t['top1'])} |",
                f"| Top-5 accuracy | {pct(t['top5'])} |",
                f"| Mean / median candidates | {t['mean_candidates']:.2f} / {t['median_candidates']} |",
                f"| Words in the word list | {t['word_list_words']} |",
                f"| Corpus tokens / bigrams | {t['corpus_tokens']} / {t['bigrams']} |", ""]

    ph = read(os.path.join(RUNS, "phrases.json"))
    if ph:
        p = ph["summary"]
        out += ["## Supplementary phrase and sentence images", "", "| Metric | Result |", "|---|---|",
                f"| Images | {p['images']} |", f"| Word error rate | {p['word_error_rate']}% |",
                f"| Transliteration CER | {p['transliteration_cer']}% |", f"| Exact match | {p['exact_match']}% |", ""]

    params = read(os.path.join(MODELS, "pipeline_params.json"))
    if params:
        out += ["## Settings tuned on the development set", "", "```json", json.dumps(params, indent=2), "```", ""]

    out += ["English translations are graded by researchers in runs/english_grading_*.csv.", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
