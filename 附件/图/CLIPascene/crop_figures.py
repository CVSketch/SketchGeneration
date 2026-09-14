#!/usr/bin/env python3
"""Re-crop CLIPascene figures from _pages/pN-NN.png (pdftoppm -png -r 200)."""
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
PAGES = BASE / "_pages"

# (page, x0, y0, x1, y1) — pixels on 200 DPI page PNG
CROPS = {
    "fig1.png": (1, 120, 508, 1580, 1165),
    "fig2.png": (2, 138, 205, 847, 545),
    "fig3.png": (2, 138, 555, 847, 1078),
    "fig4.png": (2, 852, 205, 1561, 485),
    "fig5.png": (2, 852, 498, 1561, 835),
    "fig6.png": (3, 138, 128, 847, 542),
    "fig7.png": (4, 852, 188, 1561, 672),
    "fig8.png": (5, 138, 95, 847, 515),
    "fig9.png": (5, 852, 95, 1561, 832),
    "fig10.png": (6, 138, 145, 847, 655),
    "fig11.png": (6, 852, 88, 1561, 848),
    "fig12.png": (7, 138, 88, 847, 588),
    "fig13.png": (7, 852, 88, 1561, 822),
    "fig14.png": (8, 50, 88, 1650, 775),
    "table1.png": (8, 138, 828, 847, 1172),
    "table2.png": (9, 115, 88, 847, 512),
}


def page_path(n: int) -> Path:
    return PAGES / f"p{n}-{n:02d}.png"


def main() -> None:
    for name, (page, x0, y0, x1, y1) in CROPS.items():
        src = page_path(page)
        if not src.is_file():
            raise SystemExit(f"missing page render: {src}")
        im = Image.open(src)
        im.crop((x0, y0, x1, y1)).save(BASE / name)
        print(f"wrote {name} from page {page}")


if __name__ == "__main__":
    main()
