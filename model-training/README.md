# BaybAI model (paper method)

Implements the methodology of *BaybAI: Bringing Translation of Baybayin to
Tagalog to English Through OCR* (Sections 3.2–3.6), built on the original
project's code. Only methods stated in the paper are used.

```
image ─► grayscale ─► median filter ─► Otsu ─► minor deskew ─► crop to text
      ─► connected components ─► words (horizontal gap > threshold)
      ─► kudlit attached to the base character it horizontally overlaps
      ─► each base character centred into 28×28 binary
      ─► CNN: 17 base classes            (SVM trained on the same data = baseline)
      ─► diacritic classifier: kudlit above / kudlit below / virama
      ─► all Latin readings (e/i, o/u, da/ra) ─► keep those in the Tagalog word list
      ─► rank by corpus frequency (+ neighbouring word for phrases)
         none left ─► most probable reading, marked unverified
      ─► English: dictionary (single word) / pretrained NMT, NLLB-200 (phrases)
      ─► secondary check: OpenAI vision model ("Luna AI", baybayin/verify.py)
```

## Files

| File | Paper section | Purpose |
|---|---|---|
| `baybayin_preprocess.py` | 3.5 | preprocessing, segmentation, 28×28 normalisation, character dataset |
| `train_models.py` | 3.5 | base-character CNN (17 classes) and SVM baseline on the same data |
| `train_diacritics.py` | 3.5 | diacritic classifier (above / below / virama) |
| `translate_pipeline.py` | 3.5 | readings, word-list matching, ranking, English, oracle self-test |
| `evaluate.py` | 3.2, 3.4, 3.6 | inclusion criteria, 20/80 dev/test split, tuning, metrics |
| `serve.py` | — | website (`http://localhost:8000`) |
| `baybayin/verify.py` | 5 | secondary AI check (needs `OPENAI_API_KEY` in the project `.env`) |
| `data/sentence_transcriptions.csv` | 3.2 | supplementary phrase set ground truth |
| `data/word_image_exclusions.csv` | 3.2 | images researchers exclude (criteria 1–3), with reason |
| `data/word_image_confirmed_words.csv` | 3.2 | words researchers confirm as Tagalog (criterion 4) |

## Data

- **Characters (training):** `Baybayin Symbol Dataset` → 36,000 28×28 images,
  group-disjoint split 23,026 / 5,798 / 7,176 (`processed-data/paper_split/`,
  identical to the dataset behind Table 1). Rebuild with
  `python baybayin_preprocess.py dataset ../assets --out ../processed-data/paper_split`.
- **Diacritics (training):** `Baybayin Handwritten Images` (characters with
  kudlit / virama, writer-disjoint split).
- **Word images (testing only):** `Baybayin Word Images`, filtered by the four
  inclusion criteria, 20% development / 80% test (fixed seed).
- **Supplementary phrases (testing only):** sentence photos with
  `data/sentence_transcriptions.csv` and `Baybayin Reference Images/*.txt`.
- **Language:** `tagalog words.csv` (word list), the Tagalog side of
  `Tagalog to English.csv` (frequency corpus), Tagalog–English dictionaries.

## Run

```powershell
python train_models.py                 # CNN + SVM (all training glyphs)  -> models/
python train_diacritics.py             # diacritic classifier             -> models/
python evaluate.py split               # inclusion criteria + dev/test split
python evaluate.py tune                # thresholds on the development set
python evaluate.py test                # final word-image evaluation
python evaluate.py phrases             # supplementary phrase set
python translate_pipeline.py selftest  # oracle candidate test (Table 4)
python translate_pipeline.py image path\to\image.jpg
python serve.py                        # website
python -m unittest discover -s tests -t .
```

Phrase translation uses NLLB-200 through `transformers`
(`pip install transformers sentencepiece`, ~2.4 GB model download on first
use). Without it, phrases fall back to dictionary glosses and the output says so.
English results are graded by researchers in `runs/english_grading_*.csv`.
