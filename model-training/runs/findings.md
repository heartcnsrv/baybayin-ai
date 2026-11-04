# Findings

## Table 1. CNN and SVM character classification (paper test split)

| Metric | CNN | SVM |
|---|---|---|
| Classification accuracy | 99.02% | 95.47% |
| Macro-averaged F1-score | 0.9896 | 0.9526 |

## Supporting classifiers

| Classifier | Test accuracy | Macro-F1 | Note |
|---|---|---|---|
| Diacritic CNN (above / below / virama) | 95.05% | 0.9472 | 364 marks, unseen writers |
| Script SVM (Baybayin vs. not) | 98.35% | 0.9615 | Baybayin kept 99.52%, non-Baybayin rejected 90.23% |

## Table 2. Word images, held-out test set

| Metric | Result |
|---|---|
| Images tested | 732 |
| Character accuracy | 79.71% |
| Character error rate (CER) | 22.61% |
| Word error rate (WER) | 58.88% |
| Transliteration exact match | 48.09% |
| Tagalog identification accuracy | 44.81% |
| Transliteration correct, word wrong | 24 |

Errors by earliest stage:

| Stage | Images |
|---|---|
| correct | 328 |
| kudlit / virama | 249 |
| character recognition | 109 |
| Tagalog word identification | 24 |
| unclear image (nothing segmented) | 18 |
| transliteration (word segmentation) | 4 |

## Table 4. Candidate generation, oracle self-test

| Metric | Result |
|---|---|
| Words tested | 2000 |
| Candidate recall | 100.00% |
| Top-1 accuracy | 96.90% |
| Top-5 accuracy | 100.00% |
| Mean / median candidates | 1.05 / 1.0 |
| Words in the word list | 39419 |
| Corpus tokens / bigrams | 5965543 / 164844 |

## Supplementary phrase and sentence images

| Metric | Result |
|---|---|
| Images | 37 |
| Word error rate | 104.31% |
| Transliteration CER | 63.41% |
| Exact match | 0.0% |

## Settings tuned on the development set

```json
{
  "preprocess": {
    "mark_ratio": 0.4,
    "word_gap_ratio": 1.0
  },
  "tuned_on": "development set",
  "development_result": {
    "images": 183,
    "character_accuracy": 83.28,
    "character_error_rate": 18.63,
    "word_error_rate": 54.1,
    "transliteration_exact_match": 53.01,
    "tagalog_identification_accuracy": 49.73,
    "transliteration_correct_but_word_wrong": 6,
    "errors_by_earliest_stage": {
      "correct": 91,
      "kudlit / virama": 64,
      "character recognition": 21,
      "Tagalog word identification": 6,
      "transliteration (word segmentation)": 1
    }
  },
  "script_min_share": 0.1,
  "script_tuning": {
    "baybayin_kept": 0.9781420765027322,
    "latin_rejected": 0.8761904761904762,
    "data": "development word images + validation Latin pages"
  }
}
```

English translations are graded by researchers in runs/english_grading_*.csv.
