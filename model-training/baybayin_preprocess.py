"""
baybayin_preprocess.py
======================
Preprocessing and segmentation for the Baybayin OCR transliteration translator,
following the paper's Section 3.5 (based on the original project's module).

Pipeline
--------
  image
    -> grayscale -> median filter (noise) -> Otsu binarisation
    -> minor skew correction -> crop to the region containing the text
    -> connected-component labelling (each separate group of strokes)
    -> components split into base characters / kudlit-virama marks / noise
    -> bases ordered into lines, left to right; words separated where the
       horizontal gap is wider than a threshold (tuned on the development set)
    -> each mark assigned to the base character it horizontally overlaps
    -> every base character resized and centred into a 28x28 binary image

Output polarity: ink = 255, background = 0. The SAME normalisation is used for
training data and inference.

Usage
-----
  python baybayin_preprocess.py char     path/to/char.jpg       --out out/
  python baybayin_preprocess.py image    path/to/image.png      --out out/
  python baybayin_preprocess.py dataset  ../assets              --out ../processed-data/paper_split
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GLYPH_SIZE = 28


# --------------------------------------------------------------------------- #
# Parameters (segmentation thresholds are tuned on the development set)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    median_ksize: int = 3            # median filter size (odd)
    deskew_max_angle: float = 10.0   # minor skew only: degrees searched either side
    crop_pad: int = 6                # padding kept around the text crop
    # component classification (areas relative to a reference base area)
    min_abs_area: int = 6            # components smaller than this are noise
    noise_ratio: float = 0.01        # < noise_ratio * ref_area  -> noise
    mark_ratio: float = 0.30         # < mark_ratio  * ref_area  -> kudlit/virama
    # grouping
    line_split_ratio: float = 0.6    # new line if |dy| > ratio * median base height
    word_gap_ratio: float = 0.6      # new word if gap > ratio * median base width
    # glyph normalisation
    glyph_margin: float = 0.12       # fraction of side kept as blank border


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Comp:
    idx: int
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class Mark:
    bbox: Tuple[int, int, int, int]
    image: np.ndarray            # 28x28 binary: the mark together with its base character,
                                 # input of the diacritic classifier (above / below / virama)


@dataclass
class Glyph:
    bbox: Tuple[int, int, int, int]
    image: np.ndarray            # 28x28 binary, base character only
    marks: List[Mark] = field(default_factory=list)


@dataclass
class Word:
    glyphs: List[Glyph]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _union([g.bbox for g in self.glyphs] +
                      [m.bbox for g in self.glyphs for m in g.marks])


@dataclass
class Line:
    words: List[Word]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _union([w.bbox for w in self.words])


@dataclass
class Result:
    binary: np.ndarray
    lines: List[Line]
    orphan_marks: List[Tuple[int, int, int, int]]
    noise_count: int
    offset: Tuple[int, int] = (0, 0)   # crop offset inside the deskewed image
    angle: float = 0.0                 # deskew rotation applied (degrees)
    src_shape: Tuple[int, int] = (0, 0)  # (height, width) of the input image

    def to_source_box(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Map an (x, y, w, h) box from the processed image back onto the input
        image: undo the crop offset, then undo the deskew rotation."""
        x, y, w, h = box
        ox, oy = self.offset
        pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32) + (ox, oy)
        H, W = self.src_shape
        if self.angle:
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -self.angle, 1.0)
            pts = np.hstack([pts, np.ones((4, 1), np.float32)]) @ M.T
        x0, y0 = np.clip(pts.min(0), 0, [W - 1, H - 1])
        x1, y1 = np.clip(pts.max(0), 0, [W - 1, H - 1])
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    def to_dict(self) -> Dict:
        """JSON-serialisable summary (no images)."""
        return {
            "noise_components": self.noise_count,
            "orphan_marks": [list(b) for b in self.orphan_marks],
            "lines": [{
                "bbox": list(l.bbox),
                "words": [{
                    "bbox": list(w.bbox),
                    "glyphs": [{"bbox": list(g.bbox), "marks": [list(m.bbox) for m in g.marks]}
                               for g in w.glyphs],
                } for w in l.words],
            } for l in self.lines],
        }


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Load as 3-channel BGR (works with non-ASCII Windows paths).
    Transparent PNGs are composited on white."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


# --------------------------------------------------------------------------- #
# Binarisation, deskew, crop
# --------------------------------------------------------------------------- #
def binarize(img: np.ndarray, p: Params, light_text: Optional[bool] = None) -> np.ndarray:
    """Grayscale -> median filter -> Otsu. Ink (text) = 255, background = 0.

    Otsu splits the image into a darker and a lighter class; which one is the
    text depends on the image. light_text=None guesses (text is the smaller
    class); True/False forces light text on darker background / dark text."""
    gray = cv2.medianBlur(_gray(img), _odd(p.median_ksize))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if light_text is None:
        light_text = np.count_nonzero(bw) > 0.5 * bw.size
    return cv2.bitwise_not(bw) if light_text else bw


def deskew(bw: np.ndarray, max_angle: float = 10.0) -> Tuple[np.ndarray, float]:
    """Correct minor skew: the angle whose row-sum profile is sharpest."""
    ys, xs = np.nonzero(bw)
    if len(xs) < 50:
        return bw, 0.0
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    if w < 1.5 * h:
        return bw, 0.0

    scale = 256.0 / max(bw.shape)
    small = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else bw
    sh, sw = small.shape
    centre = (sw / 2.0, sh / 2.0)

    best_angle, best_score = 0.0, -1.0
    for ang in np.arange(-max_angle, max_angle + 0.01, 0.5):
        M = cv2.getRotationMatrix2D(centre, float(ang), 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(ang)

    if abs(best_angle) < 0.25:
        return bw, 0.0
    H, W = bw.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), best_angle, 1.0)
    rot = cv2.warpAffine(bw, M, (W, H), flags=cv2.INTER_LINEAR)
    _, rot = cv2.threshold(rot, 127, 255, cv2.THRESH_BINARY)
    return rot, best_angle


def crop_to_content(bw: np.ndarray, pad: int = 6) -> Tuple[np.ndarray, Tuple[int, int]]:
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        return bw, (0, 0)
    H, W = bw.shape
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    return bw[y0:y1, x0:x1], (x0, y0)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize_glyph(mask: np.ndarray, size: int = GLYPH_SIZE,
                    margin: float = 0.12) -> np.ndarray:
    """Resize and centre a binary mask into a size x size binary image."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)
    crop = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.uint8) * 255
    h, w = crop.shape
    side = max(h, w)
    border = int(round(side * margin))
    canvas = np.zeros((side + 2 * border, side + 2 * border), np.uint8)
    oy = border + (side - h) // 2
    ox = border + (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    small = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return ((small >= 80).astype(np.uint8)) * 255


def preprocess_character_image(path_or_img, p: Optional[Params] = None) -> np.ndarray:
    p = p or Params()
    img = load_image(path_or_img) if isinstance(path_or_img, str) else path_or_img
    bw = binarize(img, p)
    return normalize_glyph(bw, GLYPH_SIZE, p.glyph_margin)


# --------------------------------------------------------------------------- #
# Connected components & segmentation
# --------------------------------------------------------------------------- #
def _components(bw: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = [Comp(i, int(stats[i][0]), int(stats[i][1]), int(stats[i][2]),
                  int(stats[i][3]), int(stats[i][4])) for i in range(1, n)]
    return comps, labels


def _classify(comps: List[Comp], p: Params):
    """Base characters vs small marks vs noise, by area relative to typical bases."""
    if not comps:
        return [], [], []
    areas = np.array([c.area for c in comps], float)
    p75 = np.percentile(areas, 75)
    ref = float(np.median(areas[areas >= 0.5 * p75]))
    bases, marks, noise = [], [], []
    for c in comps:
        if c.area < max(p.min_abs_area, p.noise_ratio * ref):
            noise.append(c)
        elif c.area < p.mark_ratio * ref:
            marks.append(c)
        else:
            bases.append(c)
    return bases, marks, noise


def _group_lines(bases: List[Comp], p: Params) -> List[List[Comp]]:
    """Order base characters: top-to-bottom lines, left-to-right within a line."""
    if not bases:
        return []
    thr = p.line_split_ratio * float(np.median([b.h for b in bases]))
    lines: List[List[Comp]] = []
    for b in sorted(bases, key=lambda c: c.cy):
        if lines and abs(b.cy - np.mean([c.cy for c in lines[-1]])) <= thr:
            lines[-1].append(b)
        else:
            lines.append([b])
    lines.sort(key=lambda l: np.mean([c.cy for c in l]))
    return [sorted(l, key=lambda c: c.x) for l in lines]


def _assign_marks(lines: List[List[Comp]], marks: List[Comp]):
    """Each mark goes to the nearest base character it horizontally overlaps
    (nearest by vertical distance among overlapping bases). Marks overlapping no
    base are kept as orphans."""
    per_line: List[Dict[int, List[Comp]]] = [dict() for _ in lines]
    orphans: List[Comp] = []
    for m in marks:
        best, best_d, best_li = None, None, None
        for li, bases in enumerate(lines):
            for b in bases:
                ov = min(m.x + m.w, b.x + b.w) - max(m.x, b.x)
                if ov <= 0:
                    continue
                d = abs(m.cy - b.cy)
                if best_d is None or d < best_d:
                    best, best_d, best_li = b, d, li
        if best is None:
            orphans.append(m)
        else:
            per_line[best_li].setdefault(best.idx, []).append(m)
    return per_line, orphans


def _split_words(bases: List[Comp], p: Params) -> List[List[Comp]]:
    """New word where the horizontal gap exceeds word_gap_ratio x median width."""
    if not bases:
        return []
    med_w = float(np.median([b.w for b in bases]))
    words, cur = [], [bases[0]]
    for prev, nxt in zip(bases, bases[1:]):
        gap = nxt.x - (prev.x + prev.w)
        if gap > p.word_gap_ratio * med_w:
            words.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    words.append(cur)
    return words


def _mark_context(labels: np.ndarray, base: Comp, mark: Comp, p: Params) -> np.ndarray:
    """The mark and its base character together, normalised to 28x28, so the
    diacritic classifier sees the mark's shape and where it sits."""
    x, y, w, h = _union([base.bbox, mark.bbox])
    region = labels[y:y + h, x:x + w]
    mask = (region == base.idx) | (region == mark.idx)
    return normalize_glyph(mask, GLYPH_SIZE, p.glyph_margin)


def segment(bw: np.ndarray, p: Optional[Params] = None, single_word: bool = False) -> Result:
    p = p or Params()
    comps, labels = _components(bw)
    bases, marks, noise = _classify(comps, p)
    if single_word and bases:
        line_groups = [sorted(bases, key=lambda b: b.x)]   # a word image holds one word
    else:
        line_groups = _group_lines(bases, p)
    per_line_marks, orphans = _assign_marks(line_groups, marks)

    lines: List[Line] = []
    for li, group in enumerate(line_groups):
        word_groups = [group] if single_word else _split_words(group, p)
        words = []
        for wg in word_groups:
            glyphs = []
            for b in wg:
                mask = (labels[b.y:b.y + b.h, b.x:b.x + b.w] == b.idx)
                g = Glyph(bbox=b.bbox, image=normalize_glyph(mask, GLYPH_SIZE, p.glyph_margin))
                for m in per_line_marks[li].get(b.idx, []):
                    g.marks.append(Mark(bbox=m.bbox, image=_mark_context(labels, b, m, p)))
                glyphs.append(g)
            words.append(Word(glyphs))
        lines.append(Line(words))
    return Result(bw, lines, [m.bbox for m in orphans], len(noise))


def process_image(img, p: Optional[Params] = None, single_word: bool = False,
                  light_text: Optional[bool] = None) -> Result:
    """Full Section 3.5 preprocessing + segmentation of one image."""
    p = p or Params()
    if isinstance(img, str):
        img = load_image(img)
    bw = binarize(img, p, light_text)
    bw, angle = deskew(bw, p.deskew_max_angle)
    bw, offset = crop_to_content(bw, p.crop_pad)
    res = segment(bw, p, single_word=single_word)
    res.offset, res.angle, res.src_shape = offset, angle, img.shape[:2]
    return res


# --------------------------------------------------------------------------- #
# Visualisation & saving
# --------------------------------------------------------------------------- #
def draw_overlay(res: Result) -> np.ndarray:
    vis = 255 - cv2.cvtColor(res.binary, cv2.COLOR_GRAY2BGR)
    for line in res.lines:
        for word in line.words:
            x, y, w, h = word.bbox
            cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 200, 255), 1)
            for g in word.glyphs:
                gx, gy, gw, gh = g.bbox
                cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 170, 0), 1)
                for m in g.marks:
                    mx, my, mw, mh = m.bbox
                    cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 0, 220), 1)
    for (x, y, w, h) in res.orphan_marks:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 0, 200), 1)
    return vis


def save_outputs(res: Result, out_dir: str, stem: str = "result") -> None:
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_binary.png"), 255 - res.binary)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), draw_overlay(res))
    with open(os.path.join(out_dir, f"{stem}_segments.json"), "w") as f:
        json.dump(res.to_dict(), f, indent=2)
    for li, line in enumerate(res.lines):
        for wi, word in enumerate(line.words):
            for gi, g in enumerate(word.glyphs):
                base = os.path.join(out_dir, f"{stem}_L{li}_W{wi}_G{gi}")
                cv2.imwrite(base + ".png", g.image)
                for mi, m in enumerate(g.marks):
                    cv2.imwrite(f"{base}_mark{mi}.png", m.image)


# --------------------------------------------------------------------------- #
# Character dataset (the labelled isolated characters used for training)
# --------------------------------------------------------------------------- #
def process_symbol_dataset(dataset_root: str, out_dir: str) -> None:
    """Baybayin Symbol Dataset -> symbols_dataset.npz (X 28x28, y class, groups source)."""
    symbols_dir = os.path.join(dataset_root, "Baybayin Symbol Dataset")
    os.makedirs(out_dir, exist_ok=True)
    X, y, groups = [], [], []
    for label in sorted(os.listdir(symbols_dir)):
        class_path = os.path.join(symbols_dir, label)
        if not os.path.isdir(class_path):
            continue
        for fn in sorted(os.listdir(class_path)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                X.append(preprocess_character_image(os.path.join(class_path, fn)))
                y.append(label)
                # Names encode a source image (e.g. a_00001_file001.jpg); kept so the
                # test split can keep every source group on one side.
                groups.append(os.path.splitext(fn)[0].rsplit("_", 1)[-1])
    out = os.path.join(out_dir, "symbols_dataset.npz")
    np.savez_compressed(out, X=np.array(X, np.uint8), y=np.array(y), groups=np.array(groups))
    print(f"{len(X)} symbol images -> {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["char", "image", "dataset"])
    ap.add_argument("path", help="image file, or the assets folder for 'dataset'")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--single-word", action="store_true", help="the image holds one word")
    args = ap.parse_args()

    if args.kind == "dataset":
        process_symbol_dataset(args.path, args.out)
        return
    stem = os.path.splitext(os.path.basename(args.path))[0]
    if args.kind == "char":
        g = preprocess_character_image(args.path, Params())
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, f"{stem}_28.png"), g)
        print("saved", os.path.join(args.out, f"{stem}_28.png"))
        return
    res = process_image(args.path, single_word=args.single_word)
    save_outputs(res, args.out, stem)
    n_words = sum(len(l.words) for l in res.lines)
    n_glyphs = sum(len(w.glyphs) for l in res.lines for w in l.words)
    n_marks = sum(len(g.marks) for l in res.lines for w in l.words for g in w.glyphs)
    print(f"{len(res.lines)} lines, {n_words} words, {n_glyphs} base characters, {n_marks} marks, "
          f"{len(res.orphan_marks)} orphan marks, {res.noise_count} noise components")


if __name__ == "__main__":
    _main()
