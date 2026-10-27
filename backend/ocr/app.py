import hmac
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from baybayin_preprocess import Params, process_image


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
CHARACTER_GLYPHS = {
    "a": "ᜀ", "ei": "ᜁ", "ou": "ᜂ", "ka": "ᜃ", "ga": "ᜄ",
    "nga": "ᜅ", "ta": "ᜆ", "dara": "ᜇ", "na": "ᜈ", "pa": "ᜉ",
    "ba": "ᜊ", "ma": "ᜋ", "ya": "ᜌ", "la": "ᜎ", "wa": "ᜏ",
    "sa": "ᜐ", "ha": "ᜑ",
}
CONSONANTS = {"ka", "ga", "nga", "ta", "dara", "na", "pa", "ba", "ma", "ya", "la", "wa", "sa", "ha"}


def build_cnn(n_classes: int) -> torch.nn.Module:
    def block(cin: int, cout: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(cout, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

    class SmallCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(block(1, 32), block(32, 64))
            self.head = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Dropout(0.4),
                torch.nn.Linear(64 * 7 * 7, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.features(x))

    return SmallCNN()


def load_model() -> tuple[torch.nn.Module, list[str]]:
    # weights_only prevents arbitrary Python objects in the supplied checkpoint
    # from being deserialized.
    checkpoint = torch.load(ROOT / "models" / "cnn.pt", map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = build_cnn(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


torch.set_num_threads(1)
MODEL, CLASSES = load_model()
app = FastAPI(title="Baybayin OCR", docs_url=None, redoc_url=None)


def _glyph_text(label: str, marks: list) -> tuple[str, str]:
    glyph = CHARACTER_GLYPHS.get(label)
    if glyph is None:
        return "", ""
    if label not in CONSONANTS:
        return glyph, {"a": "a", "ei": "i/e", "ou": "u/o"}.get(label, "")

    mark = max(marks, key=lambda item: item.bbox[2] * item.bbox[3]) if marks else None
    if not mark:
        return glyph, label.replace("dara", "da/ra")
    if mark.shape_hint == "cross":
        return glyph + "᜔", label.replace("dara", "d/r")
    if mark.position == "above":
        return glyph + "ᜒ", label.replace("dara", "d/r") + "i/e"
    if mark.position == "below":
        return glyph + "ᜓ", label.replace("dara", "d/r") + "u/o"
    return glyph, label.replace("dara", "da/ra")


def recognize(image: np.ndarray, mode: str) -> dict:
    params = Params(min_line_glyphs=1, line_split_ratio=1.2, adaptive_word_gap=True) if mode == "photo" else None
    segmented = process_image(image, mode=mode, p=params)
    output_lines = []
    confidences = []

    for line in segmented.lines:
        if line.script != "baybayin" and mode != "photo":
            continue
        output_words = []
        reading_words = []
        for word in line.words:
            if not word.glyphs:
                continue
            batch = np.stack([glyph.image for glyph in word.glyphs]).astype(np.float32) / 255.0
            tensor = torch.from_numpy(batch[:, None, :, :])
            with torch.inference_mode():
                probabilities = torch.softmax(MODEL(tensor), dim=1).cpu().numpy()
            recognized_glyphs = []
            reading_glyphs = []
            for prediction, glyph in zip(probabilities, word.glyphs):
                index = int(prediction.argmax())
                label = CLASSES[index]
                confidence = float(prediction[index])
                confidences.append(confidence)
                baybayin, reading = _glyph_text(label, glyph.marks)
                recognized_glyphs.append(baybayin)
                reading_glyphs.append(reading)
            output_words.append("".join(recognized_glyphs))
            reading_words.append("".join(reading_glyphs))
        if output_words:
            output_lines.append({"baybayin": " ".join(output_words), "reading": " ".join(reading_words)})

    if not output_lines:
        raise HTTPException(status_code=422, detail="No Baybayin characters were detected. Try a clearer, closer image.")

    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "baybayin": "\n".join(line["baybayin"] for line in output_lines),
        "reading": "\n".join(line["reading"] for line in output_lines),
        "confidence": round(confidence, 3),
        "needsReview": confidence < 0.65,
        "note": "OCR may confuse similar glyphs or vowel marks. Review the recognized text." if confidence < 0.65 else "Review the recognized text before relying on it.",
    }


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model": "baybayin-cnn"}


@app.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    mode: str = Form("photo"),
    authorization: str | None = Header(default=None),
) -> dict:
    expected = os.environ.get("OCR_SERVICE_TOKEN", "")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if mode not in {"photo", "clean"}:
        raise HTTPException(status_code=400, detail="Mode must be photo or clean.")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, WebP, or BMP image.")

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 4 MB or smaller.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large. Resize it and try again.")
            source.load()
            rgb = np.asarray(source.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Could not read that image file.") from error
    decoded = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return await run_in_threadpool(recognize, decoded, mode)
