#!/usr/bin/env python3

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
SOURCE_PDF = Path("/tmp/arxivsrc/minima/N_vac_KS_scatter_all.pdf")
IMAGE = REPO / "tmp/pdfs/N_vac_KS_scatter_all_150dpi.png"
OUTDIR = REPO / "paper_benchmarks/2023_minima"
BIN_CSV = OUTDIR / "figure1_digitized_bins.csv"
SUMMARY_CSV = OUTDIR / "figure1_digitized_summary_by_h11.csv"
H4_30_CSV = OUTDIR / "figure1_digitized_h11_004_030_summary.csv"
OVERLAY = OUTDIR / "figure1_digitized_overlay.png"

# Calibrated from the rendered 150 DPI image gridlines.
X_H1 = 507.5
X_H491 = 8094.5
Y_N1 = 5916.5
Y_N54 = 380.5

# Calibrated from the colorbar body in the same render.
CBAR_X1 = 8193
CBAR_X2 = 8273
CBAR_Y_1000 = 105.0
CBAR_Y_1 = 6192.0


def x_of_h(h: int) -> float:
    return X_H1 + (h - 1) * (X_H491 - X_H1) / (491 - 1)


def y_of_nvac(nvac: int) -> float:
    return Y_N1 - (nvac - 1) * (Y_N1 - Y_N54) / (54 - 1)


def colorbar_palette(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_top = int(round(CBAR_Y_1000))
    y_bottom = int(round(CBAR_Y_1))
    colors: list[np.ndarray] = []
    values: list[float] = []
    for y in range(y_top, y_bottom + 1):
        strip = arr[y, CBAR_X1:CBAR_X2 + 1, :].astype(float)
        colors.append(np.median(strip, axis=0))
        frac = (CBAR_Y_1 - y) / (CBAR_Y_1 - CBAR_Y_1000)
        values.append(1.0 + frac * (1000.0 - 1.0))
    return np.vstack(colors), np.array(values)


def nearest_count(rgb: np.ndarray, palette: np.ndarray, values: np.ndarray) -> float:
    distances = np.sum((palette - rgb[None, :]) ** 2, axis=1)
    return float(values[int(np.argmin(distances))])


def marker_rgb(arr: np.ndarray, x: float, y: float, radius: int = 5) -> np.ndarray | None:
    xi = int(round(x))
    yi = int(round(y))
    patch = arr[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1, :].astype(int)
    if patch.size == 0:
        return None
    span = patch.max(axis=2) - patch.min(axis=2)
    bright = patch.mean(axis=2)
    mask = (span > 16) & (bright < 245)
    if int(mask.sum()) < 3:
        return None
    pixels = patch[mask].astype(float)
    return np.median(pixels, axis=0)


def weighted_median(items: list[tuple[int, float]]) -> int:
    total = sum(weight for _, weight in items)
    running = 0.0
    for nvac, weight in sorted(items):
        running += weight
        if running >= 0.5 * total:
            return nvac
    return items[-1][0]


def ensure_rendered_image() -> None:
    if IMAGE.is_file():
        return
    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"missing source PDF: {SOURCE_PDF}")
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm is required to render the figure PDF")
    IMAGE.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = str(IMAGE.with_suffix(""))
    subprocess.run(
        [pdftoppm, "-r", "150", "-png", "-singlefile", str(SOURCE_PDF), output_prefix],
        check=True,
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ensure_rendered_image()
    arr = np.array(Image.open(IMAGE).convert("RGB"))
    palette, values = colorbar_palette(arr)

    bins: list[dict[str, object]] = []
    overlay = Image.open(IMAGE).convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for h in range(1, 492):
        for nvac in range(1, 55):
            x = x_of_h(h)
            y = y_of_nvac(nvac)
            rgb = marker_rgb(arr, x, y)
            if rgb is None:
                continue
            count = nearest_count(rgb, palette, values)
            bins.append({
                "h11": h,
                "Nvac": nvac,
                "count_estimate": round(count, 2),
                "rgb_r": round(float(rgb[0]), 1),
                "rgb_g": round(float(rgb[1]), 1),
                "rgb_b": round(float(rgb[2]), 1),
                "x_px": round(x, 2),
                "y_px": round(y, 2),
            })
            xi, yi = int(round(x)), int(round(y))
            draw.rectangle((xi - 3, yi - 3, xi + 3, yi + 3), outline=(255, 0, 0))

    by_h: dict[int, list[tuple[int, float]]] = {}
    for row in bins:
        by_h.setdefault(int(row["h11"]), []).append((int(row["Nvac"]), float(row["count_estimate"])))

    totals = {h: sum(weight for _, weight in items) for h, items in by_h.items()}
    for row in bins:
        h = int(row["h11"])
        target = 1000.0 if h >= 4 else totals[h]
        row["count_normalized"] = round(float(row["count_estimate"]) * target / totals[h], 2)

    with BIN_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "h11", "Nvac", "count_estimate", "count_normalized",
            "rgb_r", "rgb_g", "rgb_b", "x_px", "y_px"
        ])
        writer.writeheader()
        writer.writerows(bins)

    summary: list[dict[str, object]] = []
    for h in sorted(by_h):
        items = by_h[h]
        total = sum(weight for _, weight in items)
        mean = sum(nvac * weight for nvac, weight in items) / total
        summary.append({
            "h11": h,
            "total_count_estimate": round(total, 2),
            "mean_Nvac_estimate": round(mean, 4),
            "median_Nvac_estimate": weighted_median(items),
            "max_Nvac_digitized": max(nvac for nvac, _ in items),
            "num_bins_digitized": len(items),
        })

    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "h11", "total_count_estimate", "mean_Nvac_estimate",
            "median_Nvac_estimate", "max_Nvac_digitized", "num_bins_digitized",
        ])
        writer.writeheader()
        writer.writerows(summary)

    with H4_30_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "h11", "total_count_estimate", "mean_Nvac_estimate",
            "median_Nvac_estimate", "max_Nvac_digitized", "num_bins_digitized",
        ])
        writer.writeheader()
        writer.writerows(row for row in summary if 4 <= int(row["h11"]) <= 30)

    overlay.save(OVERLAY)
    print(f"wrote {BIN_CSV}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {OVERLAY}")
    print(f"digitized bins: {len(bins)}")


if __name__ == "__main__":
    main()
