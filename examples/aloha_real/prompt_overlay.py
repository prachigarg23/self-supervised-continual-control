"""Clean prompt overlay: DejaVu Sans text in a tight semi-transparent dark box, top-left.

Used by raw_video_saver.py (per-frame on saved mp4s) and grid_first_frames.py.
Auto-scales font size to image height so the same call works for 480p video frames
and ~500px A4 grid cells.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap(text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        cur: list[str] = []
        for word in raw.split():
            test = " ".join(cur + [word]) if cur else word
            if font.getbbox(test)[2] <= max_w or not cur:
                cur.append(word)
            else:
                lines.append(" ".join(cur)); cur = [word]
        if cur:
            lines.append(" ".join(cur))
    return lines


def overlay_prompt(
    bgr: np.ndarray,
    prompt: str,
    *,
    margin: int = 12,
    pad: int = 8,
    alpha: float = 0.25,
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
    font_size: int | None = None,
    max_width_frac: float = 0.7,
) -> np.ndarray:
    """Return a copy of `bgr` with `prompt` rendered top-left in a tight dark box."""
    if not prompt:
        return bgr
    h, w = bgr.shape[:2]
    if font_size is None:
        font_size = max(22, int(round(h * 31 / 508)))
    font = _font(font_size)

    lines = _wrap(prompt, font, int(max_width_frac * w))
    if not lines:
        return bgr
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    line_gap = max(2, int(line_h * 0.12))
    max_tw = max(font.getbbox(l)[2] for l in lines)
    total_h = line_h * len(lines) + line_gap * (len(lines) - 1)

    x0, y0 = margin, margin
    x1, y1 = x0 + max_tw + 2 * pad, y0 + total_h + 2 * pad

    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([x0, y0, x1, y1], fill=(*fill_rgb, int(alpha * 255)))
    pil = Image.alpha_composite(pil, overlay).convert("RGB")

    d = ImageDraw.Draw(pil)
    y = y0 + pad
    for line in lines:
        d.text((x0 + pad, y), line, font=font, fill=(255, 255, 255))
        y += line_h + line_gap
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
