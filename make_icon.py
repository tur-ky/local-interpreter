#!/usr/bin/env python3
"""Generate assets/app.ico - the same glyph the app draws for its window icon."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "assets" / "app.ico"
BLUE = (47, 129, 247, 255)
WHITE = (255, 255, 255, 255)


def render(size: int) -> Image.Image:
    scale = 8                                   # supersample, then downscale
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(s * 0.03)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=int(s * 0.22), fill=BLUE)

    bars = (0.25, 0.47, 0.66, 0.47, 0.25)       # relative heights
    bar_w = s * 0.078
    gap = s * 0.062
    total = len(bars) * bar_w + (len(bars) - 1) * gap
    x = (s - total) / 2
    for h in bars:
        height = s * h
        y0 = (s - height) / 2
        d.rounded_rectangle(
            [x, y0, x + bar_w, y0 + height], radius=bar_w / 2, fill=WHITE
        )
        x += bar_w + gap
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [render(n) for n in sizes]
    imgs[-1].save(OUT, format="ICO", sizes=[(n, n) for n in sizes], append_images=imgs[:-1])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
