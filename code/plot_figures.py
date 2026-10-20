"""Reproduce the manuscript figures from the released tables and structures.

Run from the release directory::

    python code/plot_figures.py

Figures are written to ``figures/``. Figure 1 uses the VESTA renders in
``figures/vesta``. Figures 3 and S2 recompute the ideal-supercell mixing
entropy. Figure 4 recomputes like-pair shells from the EquFlash structures.
"""

from __future__ import annotations

import csv
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lattice_distortion import (  # noqa: E402
    ANIONS,
    canonical_self_images,
    periodic_images,
    prototype_from_key,
    read_poscar,
    structure_key,
)

RELEASE = CODE_DIR.parent
OUT_DIR = RELEASE / "figures"
FIGURES = OUT_DIR
DISTORTION = RELEASE / "data" / "distortion"
DESCRIPTORS = RELEASE / "data" / "descriptors"
STRUCTURES = RELEASE / "data" / "structures"
SUPERCELLS = STRUCTURES / "supercells"







fig1_VESTA = FIGURES / "vesta"

# Fewer distinct ideal bonds first, as in Figure 2 and Table 1.
fig1_PANELS = (
    {
        "letter": "(a)",
        "title": "rock salt",
        "file": "mp-22862_rock-salt.png",
    },
    {
        "letter": "(b)",
        "title": "fluorite",
        "file": "mp-20194_fluorite.png",
    },
    {
        "letter": "(c)",
        "title": "ternary spinel",
        "file": "mp-3536_Spinel-Ternary.png",
    },
    {
        "letter": "(d)",
        "title": "pyrochlore",
        "file": "mp-5373_pyrochlore.png",
    },
    {
        "letter": "(e)",
        "title": "orthorhombic perovskite",
        "file": "mp-aaacpepo_Perovskite-Orthorhombic.png",
    },
    {
        "letter": "(f)",
        "title": r"Gd$_2$Si$_2$O$_7$",
        "file": "mp-16461_Gd2Si2O7.png",
    },
)


def fig1_configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 16.5,
            "axes.labelsize": 18,
            "axes.titlesize": 18.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


def fig1_structure_rgb(path: Path) -> np.ndarray:
    """Composite one VESTA export on white and drop the corner triad."""
    image = np.array(Image.open(path).convert("RGBA"))
    alpha = image[..., 3]
    mask = alpha > 16
    labels, n_components = ndimage.label(mask)
    height, width = mask.shape
    drop = np.zeros(n_components + 1, dtype=bool)
    for index, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        rows, cols = slc
        # The a–b–c triad is a detached cluster in the lower-left corner.
        if cols.stop < int(0.28 * width) and rows.start > int(0.70 * height):
            drop[index] = True
    if drop.any():
        mask = mask.copy()
        mask[drop[labels]] = False
    rows, cols = np.where(mask)
    pad = int(0.04 * max(int(cols.max() - cols.min()), int(rows.max() - rows.min())))
    row0 = max(0, int(rows.min()) - pad)
    row1 = min(height, int(rows.max()) + 1 + pad)
    col0 = max(0, int(cols.min()) - pad)
    col1 = min(width, int(cols.max()) + 1 + pad)
    crop = image[row0:row1, col0:col1]
    crop_alpha = np.where(mask[row0:row1, col0:col1], crop[..., 3], 0)
    color = crop[..., :3].astype(np.float32)
    coverage = crop_alpha.astype(np.float32)[..., None] / 255.0
    flat = color * coverage + 255.0 * (1.0 - coverage)
    return np.clip(flat, 0, 255).astype(np.uint8)


def fig1_on_square(image: np.ndarray, margin: float = 0.11) -> np.ndarray:
    """Center the crop on a square canvas so every panel shares a frame."""
    height, width = image.shape[:2]
    side = int(np.ceil(max(height, width) * (1.0 + 2.0 * margin)))
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    row0 = (side - height) // 2
    col0 = (side - width) // 2
    canvas[row0 : row0 + height, col0 : col0 + width] = image
    return canvas


def fig1_main() -> None:
    fig1_configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.15, 7.55))
    for ax, spec in zip(axes.ravel(), fig1_PANELS, strict=True):
        ax.imshow(fig1_on_square(fig1_structure_rgb(fig1_VESTA / spec["file"])), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(spec["title"], color="0.12", pad=6)
        ax.text(
            0.02,
            0.98,
            spec["letter"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=19.5,
            fontweight="bold",
            color="0.1",
        )

    fig.subplots_adjust(left=0.012, right=0.988, top=0.945, bottom=0.02, wspace=0.02, hspace=0.16)

    pdf = OUT_DIR / "figure1.pdf"
    png = OUT_DIR / "figure1.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")









# Point sizes from figures v1/plot_figure2.py, enlarged by 50% twice.
# The 45° names are steepened so the larger type keeps the same width.
fig2_TEXT_SCALE = 1.5
fig2_FONT_SIZE = 24.75 * fig2_TEXT_SCALE
fig2_AXES_LABEL = 28.125 * fig2_TEXT_SCALE
fig2_XTICK_SIZE = 21.375 * fig2_TEXT_SCALE
fig2_YTICK_SIZE = 23.625 * fig2_TEXT_SCALE
fig2_LEGEND_SIZE = 22.5 * fig2_TEXT_SCALE
fig2_PANEL_LETTER = 33.0 * fig2_TEXT_SCALE
fig2_TICK_ROTATION = float(np.degrees(np.arccos(np.cos(np.radians(45.0)) / fig2_TEXT_SCALE)))
# Panel B omits compositions above this overall lattice distortion.
fig2_PANEL_B_MAX = 10.0

fig2_EQUFLASH_COLOR = "#009E73"
# Left to right is also the dodge at each structure.
fig2_MODELS = (
    ("equflashv2", "EquFlash v2", DISTORTION / "equflashv2.csv", fig2_EQUFLASH_COLOR),
    ("esen", "eSEN", DISTORTION / "esen.csv", "#D55E00"),
    ("mattersim", "MatterSim", DISTORTION / "mattersim.csv", "#CC79A7"),
    ("mace", "MACE", DISTORTION / "mace.csv", "#0072B2"),
    ("dpa", "DPA-3.1", DISTORTION / "dpa.csv", "#E69F00"),
)

# Fewer distinct ideal bonds on the left.
fig2_STRUCTURES = (
    ("rock-salt", "rock salt", "o", 59),
    ("fluorite", "fluorite", "s", 63),
    ("Perovskite---Cubic", "cubic perovskite", "^", 73),
    ("Perovskite---RP-Layered", "RP perovskite", "D", 80),
    ("Spinel---Ternary", "ternary spinel", "v", 105),
    ("KCoO2", r"KCoO$_2$", "P", 128),
    ("NaCoO2", r"NaCoO$_2$", "X", 143),
    ("Spinel--Binary", "binary spinel", "h", 213),
    ("pyrochlore", "pyrochlore", "p", 235),
    ("Tungsten-Bronze", "tungsten bronze", "<", 266),
    ("Perovskite--Orthorhombic", "orthorhombic perovskite", ">", 270),
    ("Gd2SiO5", r"Gd$_2$SiO$_5$", "d", 552),
    ("LaTaO4", r"LaTaO$_4$", "8", 603),
    ("Gd2Si2O7", r"Gd$_2$Si$_2$O$_7$", "H", 809),
)

fig2_MARKER_SIZE = 13.5
fig2_MARKER_EDGE = "#1A1A1A"
fig2_MARKER_EDGE_WIDTH = 1.05
fig2_BAND_COLOR = "#F3F4F6"
# Minimum bar pitch before the panel is stretched to the width of panel A.
fig2_BAR_INCH = 0.040
fig2_GROUP_GAP_INCH = 0.14
fig2_LABEL_PAD_INCH = 0.08


def fig2_configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": fig2_FONT_SIZE,
            "axes.labelsize": fig2_AXES_LABEL,
            "xtick.labelsize": fig2_XTICK_SIZE,
            "ytick.labelsize": fig2_YTICK_SIZE,
            "legend.fontsize": fig2_LEGEND_SIZE,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


def fig2_read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fig2_medians() -> dict[tuple[str, str], tuple[float, int]]:
    """Median mean_abs_pct and composition count for each model and structure."""
    values: dict[tuple[str, str], tuple[float, int]] = {}
    for key, _label, path, _color in fig2_MODELS:
        grouped: dict[str, list[float]] = {}
        for row in fig2_read_csv(path):
            grouped.setdefault(row["prototype"], []).append(float(row["mean_abs_pct"]))
        for prototype, samples in grouped.items():
            values[(key, prototype)] = (float(np.median(samples)), len(samples))
    return values


def fig2_equflash_by_structure() -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {prototype: [] for prototype, *_rest in fig2_STRUCTURES}
    for row in fig2_read_csv(fig2_MODELS[0][2]):
        value = float(row["mean_abs_pct"])
        if value > fig2_PANEL_B_MAX:
            continue
        grouped[row["prototype"]].append(value)
    return grouped


def fig2_write_table(path: Path, values: dict[tuple[str, str], tuple[float, int]]) -> None:
    fields = ["prototype", "label", "x", "n_ideal_bonds"]
    for key, _label, _path, _color in fig2_MODELS:
        fields.extend([f"n_{key}", f"median_pct_{key}"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (prototype, label, _marker, n_bonds) in enumerate(fig2_STRUCTURES):
            row: dict[str, str] = {
                "prototype": prototype,
                "label": label.replace("$", "").replace("_", ""),
                "x": str(index),
                "n_ideal_bonds": str(n_bonds),
            }
            for key, _model_label, _path, _color in fig2_MODELS:
                median, count = values.get((key, prototype), (float("nan"), 0))
                row[f"n_{key}"] = str(count)
                row[f"median_pct_{key}"] = "" if count == 0 else f"{median:.8g}"
            writer.writerow(row)


def fig2_measure_tick_labels(labels: list[str]) -> tuple[list[float], float]:
    """Horizontal and vertical extent, in inches, of the 45° structure names."""
    scratch = plt.figure(figsize=(2.0, 2.0))
    widths: list[float] = []
    heights: list[float] = []
    renderer = None
    for label in labels:
        text = scratch.text(
            0.5,
            0.5,
            label,
            fontsize=fig2_XTICK_SIZE,
            rotation=fig2_TICK_ROTATION,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        scratch.canvas.draw()
        if renderer is None:
            renderer = scratch.canvas.get_renderer()
        box = text.get_window_extent(renderer=renderer)
        widths.append(box.width / scratch.dpi)
        heights.append(box.height / scratch.dpi)
        text.remove()
    plt.close(scratch)
    return widths, max(heights)


def fig2_style_spines(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("0.25")


def fig2_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.012,
        1.015,
        letter,
        transform=ax.transAxes,
        fontsize=fig2_PANEL_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def fig2_apply_structure_ticks(ax: plt.Axes, positions: list[float] | np.ndarray) -> None:
    labels = [item[1] for item in fig2_STRUCTURES]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels, rotation=fig2_TICK_ROTATION, ha="right", va="top", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", length=4.5, pad=5)


def fig2_draw_panel_a(ax: plt.Axes, values: dict[tuple[str, str], tuple[float, int]]) -> None:
    n_structures = len(fig2_STRUCTURES)
    x = np.arange(n_structures, dtype=float)
    offsets = np.linspace(-0.36, 0.36, len(fig2_MODELS))

    for model_index, (key, _label, _path, color) in enumerate(fig2_MODELS):
        xs: list[float] = []
        ys: list[float] = []
        for structure_index, (prototype, _text, marker, _n_bonds) in enumerate(fig2_STRUCTURES):
            median, count = values.get((key, prototype), (float("nan"), 0))
            xpos = float(x[structure_index] + offsets[model_index])
            if count == 0:
                xs.append(np.nan)
                ys.append(np.nan)
                continue
            xs.append(xpos)
            ys.append(median)
            ax.plot(
                xpos,
                median,
                linestyle="none",
                marker=marker,
                markersize=fig2_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=fig2_MARKER_EDGE,
                markeredgewidth=fig2_MARKER_EDGE_WIDTH,
                zorder=4,
            )
        ax.plot(xs, ys, color=color, linewidth=1.7, solid_capstyle="round", zorder=2)

    ax.set_xlim(-0.62, n_structures - 0.38)
    ax.set_ylim(0.0, 8.0)
    ax.set_yticks([0, 2, 4, 6, 8])
    fig2_apply_structure_ticks(ax, list(x))
    ax.set_ylabel("median overall lattice distortion (%)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.88", linewidth=0.7)
    ax.tick_params(axis="y", length=4.5)
    fig2_style_spines(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=10.5,
            markerfacecolor=color,
            markeredgecolor=fig2_MARKER_EDGE,
            markeredgewidth=0.9,
            label=label,
        )
        for _key, label, _path, color in fig2_MODELS
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        handlelength=2.6,
        borderaxespad=0.35,
        labelspacing=0.35,
    )
    fig2_panel_letter(ax, "a")


def fig2_layout_panel_b(counts: list[int], label_widths: list[float]) -> tuple[list[float], list[float], float]:
    """Place each family. Return left edges, widths, and the right edge, in inches."""
    lefts: list[float] = []
    widths: list[float] = []
    prev_center: float | None = None
    right_edge = 0.0
    for index, count in enumerate(counts):
        width = count * fig2_BAR_INCH
        if prev_center is None:
            left = max(0.0, label_widths[index] - width / 2.0)
        else:
            min_center = prev_center + label_widths[index] + fig2_LABEL_PAD_INCH
            left = max(right_edge + fig2_GROUP_GAP_INCH, min_center - width / 2.0)
        lefts.append(left)
        widths.append(width)
        prev_center = left + width / 2.0
        right_edge = left + width
    return lefts, widths, right_edge + 0.22


def fig2_draw_panel_b(
    ax: plt.Axes,
    samples: dict[str, list[float]],
    label_widths: list[float],
    axes_width: float,
) -> float:
    """Draw EquFlash bars. Return the axis width actually used, in inches."""
    counts = [len(samples[prototype]) for prototype, *_rest in fig2_STRUCTURES]
    lefts, widths, content = fig2_layout_panel_b(counts, label_widths)
    scale = axes_width / content if content < axes_width else 1.0
    used = content * scale
    lefts = [value * scale for value in lefts]
    widths = [value * scale for value in widths]
    bar_pitch = fig2_BAR_INCH * scale

    for index, (prototype, _label, _marker, _n_bonds) in enumerate(fig2_STRUCTURES):
        left = lefts[index]
        width = widths[index]
        if index % 2 == 0:
            ax.axvspan(left, left + width, color=fig2_BAND_COLOR, zorder=0)
        heights = np.asarray(samples[prototype], dtype=float)
        xs = left + (np.arange(len(heights)) + 0.5) * bar_pitch
        ax.bar(
            xs,
            heights,
            width=bar_pitch * 0.86,
            color=fig2_EQUFLASH_COLOR,
            linewidth=0,
            zorder=2,
        )

    centers = [left + width / 2.0 for left, width in zip(lefts, widths)]
    fig2_apply_structure_ticks(ax, centers)
    ax.set_xlim(0.0, used)
    ax.set_ylim(0.0, fig2_PANEL_B_MAX)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("overall lattice distortion (%)")
    ax.set_axisbelow(False)
    ax.yaxis.grid(True, color="0.88", linewidth=0.7, zorder=1)
    ax.tick_params(axis="y", length=4.5)
    fig2_style_spines(ax)
    fig2_panel_letter(ax, "b")
    return used


def fig2_main() -> None:
    fig2_configure_style()
    values = fig2_medians()
    samples = fig2_equflash_by_structure()
    table_path = OUT_DIR / "figure2_overall.csv"
    fig2_write_table(table_path, values)

    labels = [item[1] for item in fig2_STRUCTURES]
    label_widths, label_height = fig2_measure_tick_labels(labels)
    # Equal category pitch has to clear the longest 45° name.
    category_span = (len(fig2_STRUCTURES) - 1) + 0.62 + 0.38
    panel_a_width = category_span * (max(label_widths) + fig2_LABEL_PAD_INCH)
    counts = [len(samples[prototype]) for prototype, *_rest in fig2_STRUCTURES]
    _lefts, _widths, panel_b_width = fig2_layout_panel_b(counts, label_widths)
    axes_width = max(panel_a_width, panel_b_width)

    left_margin = 2.45
    right_margin = 0.55
    top_margin = 1.15
    bottom_margin = label_height + 0.42
    between = label_height + 0.72
    panel_height = 5.55 * 1.5
    fig_w = left_margin + axes_width + right_margin
    fig_h = top_margin + panel_height + between + panel_height + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    def box(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    panel_b_bottom = bottom_margin
    panel_a_bottom = bottom_margin + panel_height + between
    ax_b = fig.add_axes(box(left_margin, panel_b_bottom, axes_width, panel_height))
    ax_a = fig.add_axes(box(left_margin, panel_a_bottom, axes_width, panel_height))

    fig2_draw_panel_a(ax_a, values)
    fig2_draw_panel_b(ax_b, samples, label_widths, axes_width)

    pdf = OUT_DIR / "figure2.pdf"
    png = OUT_DIR / "figure2.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    print(f"Wrote {table_path}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Figure size: {fig_w:.2f} x {fig_h:.2f} in")
    for prototype, label, _marker, _n_bonds in fig2_STRUCTURES:
        parts = []
        for key, model_label, _path, _color in fig2_MODELS:
            median, count = values.get((key, prototype), (float("nan"), 0))
            text = "—" if count == 0 else f"{median:.2f} (n={count})"
            parts.append(f"{model_label} {text}")
        print(f"{label:28s}  " + "   ".join(parts))










fig4_SUMMARY_CSV = DISTORTION / "equflashv2.csv"
fig4_SUPERCELLS = STRUCTURES / "supercells"
fig4_RELAXED_DIR = STRUCTURES / "relaxed" / "equflashv2"
# Neighbors within this of an atom's shortest like-pair distance are nearest neighbors.
fig4_NN_TOL = 0.20
fig4_CUTOFF = 6.0
# Same cutoff as Figure 2b: overall mean absolute percentage change.
fig4_DISTORTION_MAX = 10.0

fig4_COL_CC = "#CC79A7"
fig4_COL_AA = "#009E73"
# Type is doubled. The family names are rotated past 45° so that
# larger type still fits the existing category pitch.
fig4_TEXT_SCALE = 2.0
fig4_XTICK_SIZE = 16.875 * fig4_TEXT_SCALE
fig4_LABEL_SIZE = 20.25 * fig4_TEXT_SCALE
fig4_YTICK_SIZE = 16.875 * fig4_TEXT_SCALE
fig4_LEGEND_SIZE = 16.5 * fig4_TEXT_SCALE
fig4_PANEL_LETTER = 24.75 * fig4_TEXT_SCALE
fig4_TICK_ROTATION = float(np.degrees(np.arccos(np.cos(np.radians(45.0)) / fig4_TEXT_SCALE)))

# Same order and names as Figure 2. The plot label may use mathtext.
fig4_FAMILIES = (
    ("rock-salt", "rock salt", "rock salt"),
    ("fluorite", "fluorite", "fluorite"),
    ("Perovskite---Cubic", "cubic perovskite", "cubic perovskite"),
    ("Perovskite---RP-Layered", "RP perovskite", "RP perovskite"),
    ("Spinel---Ternary", "ternary spinel", "ternary spinel"),
    ("KCoO2", r"KCoO$_2$", "KCoO2"),
    ("NaCoO2", r"NaCoO$_2$", "NaCoO2"),
    ("Spinel--Binary", "binary spinel", "binary spinel"),
    ("pyrochlore", "pyrochlore", "pyrochlore"),
    ("Tungsten-Bronze", "tungsten bronze", "tungsten bronze"),
    ("Perovskite--Orthorhombic", "orthorhombic perovskite", "orthorhombic perovskite"),
    ("Gd2SiO5", r"Gd$_2$SiO$_5$", "Gd2SiO5"),
    ("LaTaO4", r"LaTaO$_4$", "LaTaO4"),
    ("Gd2Si2O7", r"Gd$_2$Si$_2$O$_7$", "Gd2Si2O7"),
)
fig4_FAMILY_INDEX = {prototype: index for index, (prototype, *_rest) in enumerate(fig4_FAMILIES)}


def fig4_configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 18 * fig4_TEXT_SCALE,
            "axes.labelsize": fig4_LABEL_SIZE,
            "axes.titlesize": fig4_LABEL_SIZE,
            "xtick.labelsize": fig4_XTICK_SIZE,
            "ytick.labelsize": fig4_YTICK_SIZE,
            "legend.fontsize": fig4_LEGEND_SIZE,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


def fig4_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.012,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=fig4_PANEL_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def fig4__pack(ideal: np.ndarray, percent: np.ndarray, absolute: np.ndarray) -> tuple[float, int, float, float] | None:
    if ideal.size == 0:
        return None
    return (
        float(ideal.mean()),
        int(ideal.size),
        float(percent.mean()),
        float(absolute.mean()),
    )


def fig4_like_pair_shells(
    lattice: np.ndarray,
    frac_ideal: np.ndarray,
    frac_relaxed: np.ndarray,
    is_anion: np.ndarray,
    cutoff: float = fig4_CUTOFF,
    tol: float = fig4_NN_TOL,
) -> dict[str, tuple[float, int, float, float] | None]:
    """Nearest like neighbors, and every other like pair inside the cutoff.

    A pair is a nearest neighbor when its ideal distance is within ``tol`` of
    the shortest like-pair distance of either atom. The returned tuple is
    (mean ideal radius, pair count, mean |Δr|/r in percent, mean |Δr| in Å).
    """
    cart_ideal = frac_ideal @ lattice
    cart_relaxed = frac_relaxed @ lattice
    n_atoms = len(frac_ideal)
    images = periodic_images(lattice, cutoff)
    shifts = images @ lattice
    self_ok = canonical_self_images(images)
    neighbor_index = np.arange(n_atoms)
    r_nn = np.full(n_atoms, np.inf)

    ideal_parts: list[np.ndarray] = []
    percent_parts: list[np.ndarray] = []
    absolute_parts: list[np.ndarray] = []
    anion_parts: list[np.ndarray] = []
    i_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []

    for start in range(0, n_atoms, 48):
        stop = min(n_atoms, start + 48)
        atom_index = np.arange(start, stop)
        dist_ideal = np.linalg.norm(
            cart_ideal[None, :, :] + shifts[:, None, :]
            - cart_ideal[start:stop, None, None, :],
            axis=-1,
        )
        keep_pair = (neighbor_index[None, None, :] > atom_index[:, None, None]) | (
            (neighbor_index[None, None, :] == atom_index[:, None, None])
            & self_ok[None, :, None]
        )
        like = is_anion[start:stop, None, None] == is_anion[None, None, :]
        mask = keep_pair & like & (dist_ideal > 1e-8) & (dist_ideal <= cutoff)
        if not np.any(mask):
            continue
        dist_relaxed = np.linalg.norm(
            cart_relaxed[None, :, :] + shifts[:, None, :]
            - cart_relaxed[start:stop, None, None, :],
            axis=-1,
        )
        chosen_ideal = dist_ideal[mask]
        chosen_relaxed = dist_relaxed[mask]
        chosen_i = np.broadcast_to(atom_index[:, None, None], dist_ideal.shape)[mask]
        chosen_j = np.broadcast_to(neighbor_index[None, None, :], dist_ideal.shape)[mask]
        delta = np.abs(chosen_relaxed - chosen_ideal)
        ideal_parts.append(chosen_ideal)
        percent_parts.append(100.0 * delta / chosen_ideal)
        absolute_parts.append(delta)
        anion_parts.append(is_anion[chosen_i])
        i_parts.append(chosen_i)
        j_parts.append(chosen_j)
        np.minimum.at(r_nn, chosen_i, chosen_ideal)
        np.minimum.at(r_nn, chosen_j, chosen_ideal)

    empty = {name: None for name in ("cc_first", "cc_higher", "aa_first", "aa_higher")}
    if not ideal_parts:
        return empty

    ideal = np.concatenate(ideal_parts)
    percent = np.concatenate(percent_parts)
    absolute = np.concatenate(absolute_parts)
    is_aa = np.concatenate(anion_parts)
    left = np.concatenate(i_parts)
    right = np.concatenate(j_parts)
    nearest = (ideal <= r_nn[left] + tol) | (ideal <= r_nn[right] + tol)

    def take(selected: np.ndarray) -> tuple[float, int, float, float] | None:
        return fig4__pack(ideal[selected], percent[selected], absolute[selected])

    return {
        "cc_first": take((~is_aa) & nearest),
        "cc_higher": take((~is_aa) & ~nearest),
        "aa_first": take(is_aa & nearest),
        "aa_higher": take(is_aa & ~nearest),
    }


def fig4__store_shell(
    record: dict[str, object],
    pair_class: str,
    shell: str,
    stats: tuple[float, int, float, float] | None,
) -> None:
    if stats is None:
        record[f"r_{pair_class}_{shell}"] = None
        record[f"n_{pair_class}_{shell}"] = 0
        record[f"pct_{pair_class}_{shell}"] = None
        record[f"abs_{pair_class}_{shell}"] = None
        return
    radius, count, percent, absolute = stats
    record[f"r_{pair_class}_{shell}"] = radius
    record[f"n_{pair_class}_{shell}"] = count
    record[f"pct_{pair_class}_{shell}"] = percent
    record[f"abs_{pair_class}_{shell}"] = absolute


def fig4_load_rows() -> list[dict[str, object]]:
    selected: dict[str, tuple[str, float]] = {}
    with fig4_SUMMARY_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            prototype = row["prototype"]
            if prototype not in fig4_FAMILY_INDEX:
                continue
            distortion = float(row["mean_abs_pct"])
            if distortion > fig4_DISTORTION_MAX:
                continue
            selected[row["structure_key"]] = (prototype, distortion)

    records: list[dict[str, object]] = []
    for index, (key, (prototype, distortion)) in enumerate(selected.items(), start=1):
        supercell = fig4_SUPERCELLS / f"{key}.vasp"
        relaxed = fig4_RELAXED_DIR / f"{key}_relaxed.vasp"
        if not supercell.is_file() or not relaxed.is_file():
            raise FileNotFoundError(f"missing structure pair for {key}")
        _lattice_ideal, symbols_ideal, frac_ideal = read_poscar(supercell)
        lattice_relaxed, symbols_relaxed, frac_relaxed = read_poscar(relaxed)
        if len(symbols_ideal) != len(symbols_relaxed) or not np.array_equal(
            symbols_ideal, symbols_relaxed
        ):
            raise ValueError(f"species mismatch: {key}")
        shells = fig4_like_pair_shells(
            lattice_relaxed,
            frac_ideal,
            frac_relaxed,
            np.isin(symbols_ideal, list(ANIONS)),
        )
        record: dict[str, object] = {
            "structure_key": key,
            "prototype": prototype,
            "family": fig4_FAMILIES[fig4_FAMILY_INDEX[prototype]][2],
            "mean_abs_pct": distortion,
        }
        for pair_class in ("cc", "aa"):
            for shell in ("first", "higher"):
                fig4__store_shell(record, pair_class, shell, shells[f"{pair_class}_{shell}"])
        records.append(record)
        if index % 50 == 0 or index == len(selected):
            print(f"  shells {index}/{len(selected)}")
    return records


def fig4_write_table(rows: list[dict[str, object]]) -> None:
    path = OUT_DIR / "figure4_shell_pct.csv"
    fields = [
        "structure_key",
        "family",
        "prototype",
        "mean_abs_pct",
        "r_cc_first",
        "n_cc_first",
        "pct_cc_first",
        "abs_cc_first",
        "r_aa_first",
        "n_aa_first",
        "pct_aa_first",
        "abs_aa_first",
        "r_cc_higher",
        "n_cc_higher",
        "pct_cc_higher",
        "abs_cc_higher",
        "r_aa_higher",
        "n_aa_higher",
        "pct_aa_higher",
        "abs_aa_higher",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields}
            for field in fields:
                value = row[field]
                if field == "mean_abs_pct" or field.startswith(("r_", "pct_", "abs_")):
                    output[field] = "" if value is None else f"{float(value):.6g}"
            writer.writerow(output)
    print(f"Wrote {path}")


def fig4_values(rows: list[dict[str, object]], prototype: str, field: str) -> np.ndarray:
    return np.array(
        [
            float(row[field])
            for row in rows
            if row["prototype"] == prototype and row[field] is not None
        ],
        dtype=float,
    )


def fig4__box_style(face: str, hatch: str | None) -> dict:
    return dict(
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "lw": 0.9},
        whiskerprops={"color": "#444444", "lw": 0.7},
        capprops={"color": "#444444", "lw": 0.7},
        boxprops={
            "facecolor": face,
            "edgecolor": "#444444",
            "lw": 0.6,
            "alpha": 0.9,
            "hatch": hatch,
        },
    )


def fig4_whisker_top(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    q1, q3 = np.percentile(samples, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    inside = samples[samples <= fence]
    return float(inside.max() if inside.size else samples.max())


def fig4_measure_tick_labels(labels: list[str]) -> tuple[list[float], float]:
    """Horizontal and vertical extent, in inches, of the 45° family names."""
    scratch = plt.figure(figsize=(2.0, 2.0))
    widths: list[float] = []
    heights: list[float] = []
    renderer = None
    for label in labels:
        text = scratch.text(
            0.5,
            0.5,
            label,
            fontsize=fig4_XTICK_SIZE,
            rotation=fig4_TICK_ROTATION,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        scratch.canvas.draw()
        if renderer is None:
            renderer = scratch.canvas.get_renderer()
        box = text.get_window_extent(renderer=renderer)
        widths.append(box.width / scratch.dpi)
        heights.append(box.height / scratch.dpi)
        text.remove()
    plt.close(scratch)
    return widths, max(heights)


def fig4_draw_shells(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    cc_field: str,
    aa_field: str,
    hatch: str | None,
    title: str,
    ylim: float,
    legend: bool = True,
) -> None:
    """Cation–cation and oxygen–oxygen boxes, one pair per oxide family."""
    offsets = {cc_field: -0.17, aa_field: 0.17}
    faces = {cc_field: fig4_COL_CC, aa_field: fig4_COL_AA}
    for prototype, *_rest in fig4_FAMILIES:
        x = fig4_FAMILY_INDEX[prototype]
        for field, offset in offsets.items():
            samples = fig4_values(rows, prototype, field)
            if samples.size == 0:
                continue
            ax.boxplot(
                samples,
                positions=[x + offset],
                **fig4__box_style(faces[field], hatch),
            )
    labels = [label for _prototype, label, _plain in fig4_FAMILIES]
    ticks = list(range(len(fig4_FAMILIES)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        labels,
        fontsize=fig4_XTICK_SIZE,
        rotation=fig4_TICK_ROTATION,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=4.0, pad=6)
    ax.set_xlim(-0.70, len(fig4_FAMILIES) - 0.30)
    ax.set_ylim(0.0, ylim)
    ax.set_yticks(np.arange(0, ylim + 1, 2))
    ax.set_ylabel(r"$|\Delta r|/r$ (%)")
    ax.set_title(title, pad=12, fontsize=fig4_LABEL_SIZE)
    if legend:
        ax.legend(
            handles=[
                Patch(
                    facecolor=fig4_COL_CC,
                    edgecolor="#444444",
                    hatch=hatch,
                    label="cation–cation",
                ),
                Patch(
                    facecolor=fig4_COL_AA,
                    edgecolor="#444444",
                    hatch=hatch,
                    label="oxygen–oxygen",
                ),
            ],
            loc="upper right",
            frameon=False,
            fontsize=fig4_LEGEND_SIZE,
            handlelength=1.6,
            labelspacing=0.30,
            borderpad=0.2,
        )


def fig4_shared_ylim(rows: list[dict[str, object]]) -> float:
    top = 0.0
    for prototype, *_rest in fig4_FAMILIES:
        for field in (
            "pct_cc_first",
            "pct_aa_first",
            "pct_cc_higher",
            "pct_aa_higher",
        ):
            top = max(top, fig4_whisker_top(fig4_values(rows, prototype, field)))
    padded = top * 1.06
    return float(max(10.0, np.ceil(padded / 2.0) * 2.0))


def fig4__iqr(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    q1, q3 = np.percentile(samples, [25, 75])
    return float(q3 - q1)


def fig4_report(rows: list[dict[str, object]]) -> None:
    print(f"plotted n={len(rows)}  omitted overall distortion > {fig4_DISTORTION_MAX:g}%")
    missing = [
        row["structure_key"]
        for row in rows
        if any(
            row[field] is None
            for field in (
                "pct_cc_first",
                "pct_aa_first",
                "pct_cc_higher",
                "pct_aa_higher",
            )
        )
    ]
    print(f"missing a shell: {len(missing)}")
    specs = (
        ("first %", "pct_cc_first", "pct_aa_first"),
        ("higher %", "pct_cc_higher", "pct_aa_higher"),
        ("first A", "abs_cc_first", "abs_aa_first"),
        ("higher A", "abs_cc_higher", "abs_aa_higher"),
    )
    for shell, cc_field, aa_field in specs:
        paired = [
            row
            for row in rows
            if row[cc_field] is not None and row[aa_field] is not None
        ]
        cc = np.array([float(row[cc_field]) for row in paired])
        aa = np.array([float(row[aa_field]) for row in paired])
        print(
            f"{shell:8} n={len(paired)}  anion>cation {int(np.sum(aa > cc))}/{len(paired)}  "
            f"median c–c {np.median(cc):.3g}  median a–a {np.median(aa):.3g}"
        )
        for prototype, _label, plain in fig4_FAMILIES:
            both = [row for row in paired if row["prototype"] == prototype]
            fcc = np.array([float(row[cc_field]) for row in both])
            faa = np.array([float(row[aa_field]) for row in both])
            led = int(np.sum(faa > fcc)) if both else 0
            print(
                f"  {plain:24} n={len(both):3}  anion-led {led}/{len(both)}  "
                f"med {np.median(fcc):.3g} / {np.median(faa):.3g}  "
                f"iqr { fig4__iqr(fcc):.3g} / {fig4__iqr(faa):.3g}"
            )
    print("ideal first-shell lengths, median |r_cc - r_aa|")
    for prototype, _label, plain in fig4_FAMILIES:
        both = [
            row
            for row in rows
            if row["prototype"] == prototype
            and row["r_cc_first"] is not None
            and row["r_aa_first"] is not None
        ]
        delta = np.array(
            [abs(float(row["r_cc_first"]) - float(row["r_aa_first"])) for row in both]
        )
        r_cc = np.median([float(row["r_cc_first"]) for row in both])
        r_aa = np.median([float(row["r_aa_first"]) for row in both])
        print(
            f"  {plain:24} r_cc {r_cc:.3f}  r_aa {r_aa:.3f}  "
            f"median |Δr| {np.median(delta):.4f}  n(|Δr|<0.05) {int(np.sum(delta < 0.05))}/{len(both)}"
        )


def fig4_main() -> None:
    fig4_configure_style()
    rows = fig4_load_rows()
    fig4_write_table(rows)
    fig4_report(rows)

    labels = [label for _prototype, label, _plain in fig4_FAMILIES]
    label_widths, label_height = fig4_measure_tick_labels(labels)
    # One data unit is wide enough for the longest 45° name.
    inches_per_unit = max(label_widths) + 0.15
    x_span = (len(fig4_FAMILIES) - 0.30) - (-0.70)
    axes_width = x_span * inches_per_unit
    left_margin = 2.35
    right_margin = 0.55
    top_margin = 1.15
    bottom_margin = label_height + 0.48
    between = 1.05
    panel_height = 3.55 * 1.5
    fig_w = left_margin + axes_width + right_margin
    fig_h = top_margin + panel_height + between + panel_height + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    def place(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    panel_b_bottom = bottom_margin
    panel_a_bottom = bottom_margin + panel_height + between
    ax_b = fig.add_axes(place(left_margin, panel_b_bottom, axes_width, panel_height))
    ax_a = fig.add_axes(place(left_margin, panel_a_bottom, axes_width, panel_height))

    ylim = fig4_shared_ylim(rows)
    fig4_draw_shells(
        ax_a,
        rows,
        "pct_cc_first",
        "pct_aa_first",
        None,
        "First shell",
        ylim,
    )
    # Family names are drawn once, on the lower panel.
    ax_a.tick_params(axis="x", labelbottom=False)
    fig4_draw_shells(
        ax_b,
        rows,
        "pct_cc_higher",
        "pct_aa_higher",
        "///",
        "Shells beyond the first shell",
        ylim,
        legend=False,
    )
    fig4_panel_label(ax_a, "(a)")
    fig4_panel_label(ax_b, "(b)")

    pdf = OUT_DIR / "figure4.pdf"
    png = OUT_DIR / "figure4.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Figure size: {fig_w:.2f} x {fig_h:.2f} in  ylim 0–{ylim:g}")










figs1_COL_O = "#C23B22"
figs1_COL_A = "#0072B2"
figs1_COL_B = "#E69F00"
figs1_COL_SITE_I = "#0072B2"
figs1_COL_SITE_II = "#009E73"
figs1_COL_BOND_NEUTRAL = "#C6C6C6"
figs1_COL_ARROW = "#2F2F2F"
figs1_COL_GHOST = "#A7A7A7"
figs1_COL_OCT = "#D3E2F1"
figs1_COL_OCT_EDGE = "#8AA0B2"
figs1_COL_MARK = "#1F4E79"

# Polar displacements in units of the cubic cell, [100] projection,
# polar axis along +y. Exaggerated so each sublattice leaves its site.
# Site I is the centre of the four surrounding A cations; site II is the
# centre of the four surrounding oxygens.
figs1_D_A = 0.08
figs1_D_B = 0.24
figs1_D_O_AP = -0.08
figs1_D_O_EQ = -0.15
# Cartoon shifts are exaggerated. The comparison keeps their ratio and
# reports it in angstroms.
figs1_U_I_DISPLAY = 0.09
figs1_U_SCALE_A = figs1_U_I_DISPLAY / (figs1_D_B - figs1_D_A)

# Landscape: framework on the left, site comparison on the right.
FIG_W, FIG_H = 7.15, 3.72
figs1_STRUCT_LEFT = 0.10
figs1_STRUCT_RIGHT_GAP = 0.28
figs1_RIGHT = 0.12
figs1_BOTTOM = 0.16
figs1_TOP = 0.46


def figs1_configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 12.75,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 11.25,
            "ytick.labelsize": 11.25,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


def figs1_style_blank(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figs1_fit_window(ax: plt.Axes, points: np.ndarray, footer_frac: float = 0.18) -> None:
    """Square data window with a footer strip under the cartoon."""
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = float(np.max(hi - lo))
    pad = 0.08 * span
    span_p = span + 2.0 * pad
    footer = footer_frac * span_p
    side = span_p + footer
    cx = 0.5 * float(lo[0] + hi[0])
    cy = 0.5 * float(lo[1] + hi[1])
    y_top = cy + span_p / 2.0
    ax.set_xlim(cx - side / 2.0, cx + side / 2.0)
    ax.set_ylim(y_top - side, y_top)
    ax.set_aspect("equal", adjustable="box")


def figs1_footer_text(ax: plt.Axes, x: float, y: float, text: str, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=9.75,
        color="0.25",
        clip_on=False,
        zorder=8,
    )


def figs1_arrow_between(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    *,
    rad_start: float = 0.0,
    rad_end: float = 0.0,
    lw: float = 0.75,
    zorder: int = 5,
    mutation: float = 7,
) -> None:
    vec = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vec))
    if length <= rad_start + rad_end + 0.012:
        return
    direction = vec / length
    p0 = np.asarray(start, dtype=float) + direction * rad_start
    p1 = np.asarray(end, dtype=float) - direction * rad_end
    ax.annotate(
        "",
        xy=(float(p1[0]), float(p1[1])),
        xytext=(float(p0[0]), float(p0[1])),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=mutation,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )


def figs1_perovskite_lattice(n: int = 2) -> dict[str, np.ndarray]:
    """Cubic ABO3 in [100] projection. Polar axis is y."""
    a_pts = [(float(i), float(j)) for j in range(n + 1) for i in range(n + 1)]
    b_pts = [(i + 0.5, j + 0.5) for j in range(n) for i in range(n)]
    o_ap = [(i + 0.5, float(j)) for j in range(n + 1) for i in range(n)]
    o_eq = [(float(i), j + 0.5) for j in range(n) for i in range(n + 1)]
    return {
        "A": np.asarray(a_pts, dtype=float),
        "B": np.asarray(b_pts, dtype=float),
        "Oap": np.asarray(o_ap, dtype=float),
        "Oeq": np.asarray(o_eq, dtype=float),
    }


def figs1_displaced(points: np.ndarray, dy: float) -> np.ndarray:
    out = np.array(points, dtype=float, copy=True)
    out[:, 1] += dy
    return out


def figs1_cage_center(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Mean of the four lattice neighbours of a B site."""
    delta = points - origin
    nearest = np.argsort(np.sum(delta * delta, axis=1))[:4]
    return points[nearest].mean(axis=0)


def figs1_draw_ferroelectric(ax: plt.Axes) -> tuple[float, float]:
    ideal = figs1_perovskite_lattice()
    actual = {
        "A": figs1_displaced(ideal["A"], figs1_D_A),
        "B": figs1_displaced(ideal["B"], figs1_D_B),
        "Oap": figs1_displaced(ideal["Oap"], figs1_D_O_AP),
        "Oeq": figs1_displaced(ideal["Oeq"], figs1_D_O_EQ),
    }
    # The marked B is the upper-right octahedron, away from the corner.
    mark = int(np.argmax(ideal["B"][:, 0] + ideal["B"][:, 1]))
    b_ideal = ideal["B"][mark]
    b_actual = actual["B"][mark]
    site_i = figs1_cage_center(actual["A"], b_ideal)
    site_ii = figs1_cage_center(np.vstack([actual["Oap"], actual["Oeq"]]), b_ideal)
    u_i = float(np.linalg.norm(b_actual - site_i))
    u_ii = float(np.linalg.norm(b_actual - site_ii))

    r_a, r_b, r_o = 0.070, 0.056, 0.044
    r_ghost = 0.030
    r_mark = 0.036

    for b_id, b_ac in zip(ideal["B"], actual["B"]):
        oxy = np.vstack([actual["Oap"], actual["Oeq"]])
        ring = oxy[np.argsort(np.sum((oxy - b_id) ** 2, axis=1))[:4]]
        centre = ring.mean(axis=0)
        ang = np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0])
        ring = ring[np.argsort(ang)]
        marked = float(np.linalg.norm(b_id - b_ideal)) < 1e-8
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor="#E7F0F8" if marked else figs1_COL_OCT,
                edgecolor=figs1_COL_MARK if marked else figs1_COL_OCT_EDGE,
                linewidth=1.15 if marked else 0.6,
                joinstyle="round",
                zorder=1,
            )
        )
        for o in ring:
            ax.plot(
                [b_ac[0], o[0]],
                [b_ac[1], o[1]],
                color="#D5D5D5" if marked else figs1_COL_BOND_NEUTRAL,
                lw=0.9,
                solid_capstyle="round",
                zorder=2,
            )

    groups = (
        ("A", figs1_D_A, r_a, figs1_COL_A, "#1A4E79"),
        ("B", figs1_D_B, r_b, figs1_COL_B, "#8A5A00"),
        ("Oap", figs1_D_O_AP, r_o, figs1_COL_O, "#7A1F14"),
        ("Oeq", figs1_D_O_EQ, r_o, figs1_COL_O, "#7A1F14"),
    )
    for key, _dy, radius, face, edge in groups:
        for p_id, p_ac in zip(ideal[key], actual[key]):
            skip_ghost = key == "B" and float(np.linalg.norm(p_id - b_ideal)) < 1e-8
            if skip_ghost:
                continue
            ax.add_patch(
                Circle(
                    (float(p_id[0]), float(p_id[1])),
                    r_ghost,
                    facecolor="white",
                    edgecolor=figs1_COL_GHOST,
                    linewidth=0.6,
                    zorder=3,
                )
            )
            figs1_arrow_between(
                ax,
                p_id,
                p_ac,
                figs1_COL_ARROW,
                rad_start=r_ghost * 1.05,
                rad_end=radius * 1.05,
                lw=0.7,
                zorder=4,
            )
        for index, p_ac in enumerate(actual[key]):
            marked_b = key == "B" and index == mark
            ax.add_patch(
                Circle(
                    (float(p_ac[0]), float(p_ac[1])),
                    radius,
                    facecolor=face,
                    edgecolor="#111111" if marked_b else edge,
                    linewidth=0.9 if marked_b else 0.45,
                    zorder=6,
                )
            )

    callouts = (
        (site_i, figs1_COL_SITE_I, "I", np.array([1.05, 0.22])),
        (site_ii, figs1_COL_SITE_II, "II", np.array([1.05, -0.34])),
    )
    label_pts = []
    for site, color, tag, shift in callouts:
        ax.add_patch(
            Circle(
                (float(site[0]), float(site[1])),
                r_mark,
                facecolor="white",
                edgecolor=color,
                linewidth=1.25,
                zorder=7,
            )
        )
        label_at = np.array([float(b_ideal[0]), float(site[1])]) + shift
        label_pts.append(label_at)
        ax.annotate(
            tag,
            xy=(float(site[0]) + r_mark * 0.7, float(site[1])),
            xytext=(float(label_at[0]), float(label_at[1])),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.65,
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=8,
        )

    cloud = np.vstack(
        [
            ideal["A"],
            ideal["B"],
            ideal["Oap"],
            ideal["Oeq"],
            actual["A"],
            actual["B"],
            actual["Oap"],
            actual["Oeq"],
            site_i,
            site_ii,
            *label_pts,
        ]
    )
    figs1_fit_window(ax, cloud, footer_frac=0.20)
    figs1_style_blank(ax)
    ax.set_title("Ferroelectric perovskite", pad=8, fontsize=13.5, color="0.12")

    specs = (
        (0.05, figs1_COL_A, "#1A4E79", 5.2, "A"),
        (0.20, figs1_COL_B, "#8A5A00", 5.2, "B"),
        (0.34, figs1_COL_O, "#7A1F14", 4.4, "O"),
    )
    for x, face, edge, ms, name in specs:
        ax.plot(
            [x],
            [0.055],
            "o",
            ms=ms,
            color=face,
            mec=edge,
            mew=0.4,
            transform=ax.transAxes,
            clip_on=False,
            zorder=8,
        )
        figs1_footer_text(ax, x + 0.035, 0.055, name)
    ax.annotate(
        "",
        xy=(0.62, 0.055),
        xytext=(0.52, 0.055),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=figs1_COL_ARROW, lw=0.75, mutation_scale=7),
        zorder=8,
    )
    figs1_footer_text(ax, 0.65, 0.055, "displacement")
    return u_i * figs1_U_SCALE_A, u_ii * figs1_U_SCALE_A


def figs1_draw_site_choice(ax: plt.Axes, u_i: float, u_ii: float) -> None:
    """Same B atom, two ideal sites, displacements drawn to a common scale."""
    figs1_style_blank(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ideal site is not unique", fontsize=13.5, color="0.12", pad=8)

    atom_y = 0.82
    value_y = 0.34
    name_y = 0.02
    columns = (
        (0.36, u_i, figs1_COL_SITE_I, "I", "A-sublattice\ncenter"),
        (0.74, u_ii, figs1_COL_SITE_II, "II", "oxygen-cage\ncenter"),
    )
    px_per_a = 0.40 / max(u_i, u_ii)

    ax.plot([0.24, 0.88], [atom_y, atom_y], color="#E3E3E3", lw=0.6, zorder=1)
    ax.text(0.02, atom_y, "B atom", ha="left", va="center", fontsize=9.6, color="0.45")

    for x, u_val, color, tag, name in columns:
        y_site = atom_y - u_val * px_per_a
        ax.annotate(
            "",
            xy=(x, atom_y - 0.045),
            xytext=(x, y_site + 0.034),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.4,
                mutation_scale=8,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
        ax.plot(
            [x],
            [atom_y],
            "o",
            ms=9.0,
            color=figs1_COL_B,
            mec="#8A5A00",
            mew=0.6,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
        ax.plot(
            [x],
            [y_site],
            "o",
            ms=7.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
        ax.text(
            x - 0.09,
            y_site,
            tag,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            zorder=5,
        )
        ax.text(
            x,
            value_y,
            f"|u| = {u_val:.2f} Å",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.8,
            color=color,
            zorder=5,
        )
        ax.text(
            x,
            name_y,
            name,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.9,
            color="0.28",
            linespacing=1.05,
            zorder=5,
        )


def figs1_main() -> None:
    figs1_configure_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    side = FIG_H - figs1_BOTTOM - figs1_TOP
    struct_left = figs1_STRUCT_LEFT / FIG_W
    struct_bottom = figs1_BOTTOM / FIG_H
    struct_side_x = side / FIG_W
    struct_side_y = side / FIG_H

    right_left_in = figs1_STRUCT_LEFT + side + figs1_STRUCT_RIGHT_GAP
    right_width_in = FIG_W - right_left_in - figs1_RIGHT

    ax_struct = fig.add_axes([struct_left, struct_bottom, struct_side_x, struct_side_y])
    ax_sites = fig.add_axes(
        [
            right_left_in / FIG_W,
            struct_bottom,
            right_width_in / FIG_W,
            struct_side_y,
        ]
    )

    u_i, u_ii = figs1_draw_ferroelectric(ax_struct)
    figs1_draw_site_choice(ax_sites, u_i, u_ii)

    pdf = OUT_DIR / "figure_s1.pdf"
    png = OUT_DIR / "figure_s1.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"|u|_I = {u_i:.3f} Å, |u|_II = {u_ii:.3f} Å")




# Configurational entropy is measured on the ideal supercell. A sublattice is
# a set of atoms that share the same neighbor-shell radii out to 3.3 Å.
# Anions and cations are kept apart. Vacancies are not a species.

SITE_SHELL_MAX = 3.3  # Å
SITE_SHELL_GAP = 0.15  # Å
SITE_MIN_DISTANCE = 0.35  # Å
SITE_ANIONS = {"O", "F", "N", "S", "Se", "Te", "Cl", "Br", "I", "C", "B", "P"}

FORMULA_ATOMS = {
    "rock-salt": 2,
    "fluorite": 3,
    "Perovskite---Cubic": 5,
    "Perovskite---RP-Layered": 7,
    "Spinel---Ternary": 7,
    "KCoO2": 4,
    "NaCoO2": 4,
    "Spinel--Binary": 7,
    "pyrochlore": 11,
    "Tungsten-Bronze": 9,
    "Perovskite--Orthorhombic": 5,
    "Gd2SiO5": 8,
    "LaTaO4": 6,
    "Gd2Si2O7": 11,
}
FLUORITE_CATION_ATOMS = 1
FLUORITE_ANION_ATOMS = 2

# Same order as Figure 2. The first seven families are named on panel (a).
FIG3_FAMILIES = (
    ("rock-salt", "rock salt", "o", "#0072B2"),
    ("fluorite", "fluorite", "s", "#E69F00"),
    ("Perovskite---Cubic", "cubic perovskite", "^", "#009E73"),
    ("Perovskite---RP-Layered", "RP perovskite", "D", "#56B4E9"),
    ("Spinel---Ternary", "ternary spinel", "v", "#D55E00"),
    ("KCoO2", r"KCoO$_2$", "P", "#CC79A7"),
    ("NaCoO2", r"NaCoO$_2$", "X", "#882255"),
    ("Spinel--Binary", "binary spinel", "h", "#44AA99"),
    ("pyrochlore", "pyrochlore", "p", "#332288"),
    ("Tungsten-Bronze", "tungsten bronze", "<", "#88CCEE"),
    ("Perovskite--Orthorhombic", "orthorhombic perovskite", ">", "#DDCC77"),
    ("Gd2SiO5", r"Gd$_2$SiO$_5$", "d", "#AA4499"),
    ("LaTaO4", r"LaTaO$_4$", "8", "#117733"),
    ("Gd2Si2O7", r"Gd$_2$Si$_2$O$_7$", "H", "#661100"),
)
FIG3_LEGEND_SPLIT = 7
FIG3_MAX_DISTORTION = 10.0
SMALLEST_ENVELOPES = 5


def site_anion_mask(symbols: np.ndarray) -> np.ndarray:
    present = set(symbols.tolist())
    anions = set(SITE_ANIONS)
    if "Si" in present and not (present & {"O", "F", "N", "S", "Se", "Te", "Cl", "Br", "I"}):
        anions.add("Si")
    return np.array([species in anions for species in symbols])


def site_images(cart: np.ndarray, lattice: np.ndarray, cutoff: float) -> np.ndarray:
    volume = abs(float(np.linalg.det(lattice)))
    cross = np.stack(
        [
            np.cross(lattice[1], lattice[2]),
            np.cross(lattice[2], lattice[0]),
            np.cross(lattice[0], lattice[1]),
        ]
    )
    spacings = volume / np.linalg.norm(cross, axis=1)
    n_img = np.maximum(np.ceil(cutoff / spacings).astype(int), 1)
    axes = [np.arange(-int(n), int(n) + 1) for n in n_img]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return cart[None, :, :] + (grid @ lattice)[:, None, :]


def site_shell_key(distances: np.ndarray) -> tuple[float, ...]:
    kept = np.sort(distances[(distances > SITE_MIN_DISTANCE) & (distances <= SITE_SHELL_MAX)])
    if kept.size == 0:
        return tuple()
    centers: list[float] = []
    bucket = [float(kept[0])]
    for value in kept[1:]:
        value = float(value)
        if value - bucket[-1] > SITE_SHELL_GAP:
            centers.append(round(float(np.mean(bucket)), 1))
            bucket = [value]
        else:
            bucket.append(value)
    centers.append(round(float(np.mean(bucket)), 1))
    return tuple(centers)


def site_mixing_entropy(symbols: np.ndarray, indices: list[int]) -> float:
    counts = Counter(symbols[index] for index in indices)
    fractions = np.array(list(counts.values()), dtype=float)
    fractions /= fractions.sum()
    return float(-(fractions * np.log(fractions)).sum())


def _different_occupancy(symbols: np.ndarray, left: np.ndarray, right: np.ndarray) -> bool:
    left_counts = Counter(symbols[index] for index in left)
    right_counts = Counter(symbols[index] for index in right)
    n_left = len(left)
    n_right = len(right)
    for species in set(left_counts) | set(right_counts):
        fraction_left = left_counts.get(species, 0) / n_left
        fraction_right = right_counts.get(species, 0) / n_right
        if (fraction_left >= 0.75 and fraction_right <= 0.25) or (
            fraction_right >= 0.75 and fraction_left <= 0.25
        ):
            return True
    return False


def split_bipartite(
    cart: np.ndarray,
    lattice: np.ndarray,
    symbols: np.ndarray,
    indices: list[int],
) -> list[list[int]]:
    """Split two interpenetrating sites that share one radial fingerprint."""
    if len(indices) < 24:
        return [indices]
    if len({symbols[index] for index in indices}) < 2:
        return [indices]
    points = cart[indices]
    images = site_images(points, lattice, 5.0)
    probe = np.linalg.norm(images - points[0], axis=-1).ravel()
    probe = probe[probe > SITE_MIN_DISTANCE]
    if probe.size == 0:
        return [indices]
    nearest = float(probe.min())
    limit = nearest * 1.2
    size = len(indices)
    neighbors: list[list[int]] = [[] for _ in range(size)]
    for start in range(size):
        distances = np.linalg.norm(images - points[start], axis=-1)
        hits = np.argwhere((distances > SITE_MIN_DISTANCE) & (distances <= limit))
        neighbors[start] = [int(atom) for _image, atom in hits if int(atom) != start]

    color = np.full(size, -1, dtype=int)
    bipartite = True
    for start in range(size):
        if color[start] != -1:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            current = stack.pop()
            for other in neighbors[current]:
                if color[other] == -1:
                    color[other] = 1 - color[current]
                    stack.append(other)
                elif color[other] == color[current]:
                    bipartite = False
    if not bipartite or np.any(color < 0):
        return [indices]
    chosen = np.asarray(indices)
    left = chosen[color == 0]
    right = chosen[color == 1]
    if min(len(left), len(right)) < 8:
        return [indices]
    if abs(len(left) - len(right)) / max(len(left), len(right)) > 0.15:
        return [indices]
    if not _different_occupancy(symbols, left, right):
        return [indices]
    return [left.tolist(), right.tolist()]


def sublattices(
    lattice: np.ndarray, symbols: np.ndarray, frac: np.ndarray
) -> list[dict[str, object]]:
    cart = frac @ lattice
    images = site_images(cart, lattice, SITE_SHELL_MAX + 0.6)
    anions = site_anion_mask(symbols)
    grouped: dict[tuple[bool, tuple[float, ...]], list[int]] = {}
    for index in range(len(symbols)):
        distances = np.linalg.norm(images - cart[index], axis=-1).ravel()
        key = (bool(anions[index]), site_shell_key(distances))
        grouped.setdefault(key, []).append(index)

    sites: list[dict[str, object]] = []
    for key, indices in grouped.items():
        for part in split_bipartite(cart, lattice, symbols, indices):
            counts = Counter(symbols[index] for index in part)
            sites.append(
                {
                    "n_atoms": len(part),
                    "n_elements": len(counts),
                    "entropy": site_mixing_entropy(symbols, part),
                    "is_anion": key[0],
                }
            )
    return sites


def _weighted(sites: list[dict[str, object]], atom_count: int, weight: float) -> float:
    if atom_count <= 0 or weight == 0.0:
        return 0.0
    total = 0.0
    for site in sites:
        share = int(site["n_atoms"]) / atom_count
        total += share * weight * float(site["entropy"])
    return total


def entropy_per_atom(sites: list[dict[str, object]], prototype: str, n_total: int) -> float:
    """Mixing entropy per atom of the reduced formula."""
    formula_atoms = FORMULA_ATOMS[prototype]
    if prototype == "fluorite":
        cations = [site for site in sites if not site["is_anion"]]
        anions = [site for site in sites if site["is_anion"]]
        n_cations = sum(int(site["n_atoms"]) for site in cations)
        n_anions = sum(int(site["n_atoms"]) for site in anions)
        if n_cations == 0:
            raise RuntimeError(f"{prototype}: no cation sublattice")
        return _weighted(
            cations, n_cations, FLUORITE_CATION_ATOMS / formula_atoms
        ) + _weighted(anions, n_anions, FLUORITE_ANION_ATOMS / formula_atoms)

    if n_total % formula_atoms != 0:
        raise RuntimeError(
            f"{prototype}: {n_total} atoms is not a whole number of "
            f"{formula_atoms}-atom formulas"
        )
    formulas = n_total / formula_atoms
    total = 0.0
    for site in sites:
        multiplicity = int(site["n_atoms"]) / formulas
        if abs(multiplicity - round(multiplicity)) > 1e-4:
            raise RuntimeError(
                f"{prototype}: a sublattice of {site['n_atoms']} atoms "
                f"is not an integer number per formula (Z={formulas:g})"
            )
        total += (multiplicity / formula_atoms) * float(site["entropy"])
    return total


def high_entropy_site(sites: list[dict[str, object]]) -> dict[str, object]:
    return max(sites, key=lambda site: (float(site["entropy"]), int(site["n_elements"])))


def horizontal_jitter(key: str, half_width: float = 0.17) -> float:
    """Deterministic offset that keeps an integer inside its own column."""
    spread = zlib.adler32(key.encode("utf-8")) % 1000
    return (spread / 999.0 - 0.5) * 2.0 * half_width


def _entropy_one(path_str: str) -> dict[str, object]:
    path = Path(path_str)
    lattice, symbols, frac = read_poscar(path)
    sites = sublattices(lattice, symbols, frac)
    key = structure_key(path)
    prototype = prototype_from_key(key)
    chosen = high_entropy_site(sites)
    return {
        "structure_key": key,
        "prototype": prototype,
        "n_elements": int(chosen["n_elements"]),
        "s_conf_over_r": float(sum(float(site["entropy"]) for site in sites)),
        "s_conf_per_atom": entropy_per_atom(sites, prototype, len(symbols)),
        "n_formula": FORMULA_ATOMS[prototype],
    }


def descriptor_records() -> list[dict[str, object]]:
    """Join EquFlash distortion, Shannon size mismatch, and supercell entropy."""
    with (DISTORTION / "equflashv2.csv").open(newline="", encoding="utf-8") as handle:
        distortion = {row["structure_key"]: row for row in csv.DictReader(handle)}
    mismatch: dict[str, dict[str, str]] = {}
    with (DESCRIPTORS / "atomic_size_mismatch.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mismatch[row["structure"].removesuffix("_relaxed")] = row
    missing = sorted(set(distortion) - set(mismatch))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"no atomic size mismatch for {len(missing)} structures, including {preview}")

    paths = [str(SUPERCELLS / f"{key}.vasp") for key in distortion]
    measured: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, record in enumerate(pool.map(_entropy_one, paths, chunksize=4), start=1):
            measured[str(record["structure_key"])] = record
            if index % 100 == 0 or index == len(paths):
                print(f"  entropy {index}/{len(paths)}")

    records: list[dict[str, object]] = []
    for key, row in distortion.items():
        site = measured[key]
        family = next(label for proto, label, _marker, _color in FIG3_FAMILIES if proto == row["prototype"])
        records.append(
            {
                "structure_key": key,
                "prototype": row["prototype"],
                "family": family.replace("$", "").replace("_", ""),
                "n_elements": int(site["n_elements"]),
                "n_formula": int(site["n_formula"]),
                "s_conf_over_r": float(site["s_conf_over_r"]),
                "s_conf_per_atom": float(site["s_conf_per_atom"]),
                "delta_percent": float(mismatch[key]["delta_percent"]),
                "mean_abs_pct": float(row["mean_abs_pct"]),
            }
        )
    records.sort(key=lambda item: (str(item["prototype"]), str(item["structure_key"])))
    return records


def write_descriptor_table(records: list[dict[str, object]]) -> None:
    path = OUT_DIR / "figure3_overall.csv"
    fields = [
        "structure_key",
        "prototype",
        "family",
        "n_elements",
        "n_formula",
        "s_conf_over_r",
        "s_conf_per_atom",
        "delta_percent",
        "mean_abs_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "structure_key": row["structure_key"],
                    "prototype": row["prototype"],
                    "family": row["family"],
                    "n_elements": row["n_elements"],
                    "n_formula": row["n_formula"],
                    "s_conf_over_r": f"{float(row['s_conf_over_r']):.6g}",
                    "s_conf_per_atom": f"{float(row['s_conf_per_atom']):.6g}",
                    "delta_percent": f"{float(row['delta_percent']):.6g}",
                    "mean_abs_pct": f"{float(row['mean_abs_pct']):.6g}",
                }
            )
    print(f"Wrote {path}")


def _style_descriptor_panel(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.88", linewidth=0.5)
    ax.tick_params(axis="x", length=3.5)
    ax.tick_params(axis="y", length=3.2)
    for spine in ax.spines.values():
        spine.set_color("0.25")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.02,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def envelope_polygon(xs: list[float], ys: list[float]) -> tuple[np.ndarray, float] | None:
    """Expanded convex hull and its unexpanded area. None if the hull is undefined."""
    points = np.unique(np.column_stack([xs, ys]), axis=0)
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    verts = points[hull.vertices]
    center = verts.mean(axis=0)
    expanded = center + (verts - center) * 1.10
    return expanded, float(hull.volume)


def draw_envelope(ax: plt.Axes, expanded: np.ndarray, color: str) -> None:
    ax.add_patch(
        Polygon(
            expanded,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.18,
            zorder=1,
            label="_nolegend_",
        )
    )
    closed = np.vstack([expanded, expanded[:1]])
    ax.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linewidth=1.0,
        alpha=0.9,
        zorder=2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="_nolegend_",
    )


def draw_smallest_envelopes(ax: plt.Axes, groups: list[dict[str, object]], x_key: str) -> None:
    """Keep the five families whose hulls enclose the least area on this panel."""
    ranked: list[tuple[float, np.ndarray, str]] = []
    for group in groups:
        polygon = envelope_polygon(group[x_key], group["ys"])
        if polygon is None:
            continue
        expanded, area = polygon
        ranked.append((area, expanded, str(group["color"])))
    ranked.sort(key=lambda item: item[0])
    for _area, expanded, color in ranked[:SMALLEST_ENVELOPES]:
        draw_envelope(ax, expanded, color)


def figure_3(records: list[dict[str, object]]) -> None:
    plotted = [row for row in records if float(row["mean_abs_pct"]) <= FIG3_MAX_DISTORTION]
    delta = np.array([float(row["delta_percent"]) for row in plotted])
    entropy = np.array([float(row["s_conf_per_atom"]) for row in plotted])
    distortion = np.array([float(row["mean_abs_pct"]) for row in plotted])
    rho_delta = float(spearmanr(delta, distortion).statistic)
    rho_entropy = float(spearmanr(entropy, distortion).statistic)

    mpl.rcParams.update(
        {
            "font.size": 16.5,
            "axes.labelsize": 18.75,
            "xtick.labelsize": 15.75,
            "ytick.labelsize": 15.75,
            "legend.fontsize": 12.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    panel_w = 5.9736 * 1.3
    panel_h = 3.663 * 1.3
    left = 1.70
    right = 0.28
    top = 0.62
    bottom = 1.28
    between = 1.70
    fig_w = left + panel_w + right
    fig_h = top + panel_h + between + panel_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))

    def place(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

    ax_b = fig.add_axes(place(left, bottom, panel_w, panel_h))
    ax_a = fig.add_axes(place(left, bottom + panel_h + between, panel_w, panel_h))
    style = dict(s=28, edgecolor="white", linewidths=0.35, alpha=0.85, zorder=3)
    legend_kw = dict(
        loc="upper right",
        frameon=False,
        fontsize=12.3,
        handlelength=1.15,
        labelspacing=0.30,
        borderaxespad=0.15,
        markerscale=2,
    )
    groups: list[dict[str, object]] = []
    for index, (prototype, label, marker, color) in enumerate(FIG3_FAMILIES):
        chosen = [row for row in plotted if row["prototype"] == prototype]
        if not chosen:
            continue
        ys = [float(row["mean_abs_pct"]) for row in chosen]
        groups.append(
            {
                "xs_a": [float(row["delta_percent"]) for row in chosen],
                "xs_b": [float(row["s_conf_per_atom"]) for row in chosen],
                "ys": ys,
                "color": color,
            }
        )
        common = dict(style, marker=marker, facecolor=color)
        ax_a.scatter(
            [float(row["delta_percent"]) for row in chosen],
            ys,
            label=label if index < FIG3_LEGEND_SPLIT else None,
            **common,
        )
        ax_b.scatter(
            [float(row["s_conf_per_atom"]) for row in chosen],
            ys,
            label=label if index >= FIG3_LEGEND_SPLIT else None,
            **common,
        )

    draw_smallest_envelopes(ax_a, groups, "xs_a")
    draw_smallest_envelopes(ax_b, groups, "xs_b")
    ax_a.set_xlim(0.0, 30.0)
    ax_a.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_b.set_xlim(0.0, max(0.85, float(entropy.max()) * 1.08))
    for ax in (ax_a, ax_b):
        ax.set_ylim(0.0, FIG3_MAX_DISTORTION)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        _style_descriptor_panel(ax)
        ax.set_ylabel("overall lattice distortion (%)")
    ax_a.set_xlabel("Atomic size mismatch (%)")
    ax_b.set_xlabel("Configurational entropy\nof the whole structure")
    _panel_label(ax_a, "(a)")
    _panel_label(ax_b, "(b)")
    ax_a.legend(**legend_kw)
    ax_b.legend(**legend_kw)
    pdf = OUT_DIR / "figure3.pdf"
    png = OUT_DIR / "figure3.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    omitted = len(records) - len(plotted)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(
        f"n={len(plotted)} plotted  omitted distortion>{FIG3_MAX_DISTORTION:g}%: {omitted}  "
        f"size mismatch {delta.min():.2f}–{delta.max():.2f}%  "
        f"entropy/atom {entropy.min():.3f}–{entropy.max():.3f}  "
        f"distortion {distortion.min():.2f}–{distortion.max():.2f}%  "
        f"Spearman size mismatch ρ={rho_delta:.3f}  entropy ρ={rho_entropy:.3f}"
    )


def figure_s2(records: list[dict[str, object]]) -> None:
    plotted = [row for row in records if float(row["mean_abs_pct"]) <= FIG3_MAX_DISTORTION]
    elements = np.array([int(row["n_elements"]) for row in plotted])
    distortion = np.array([float(row["mean_abs_pct"]) for row in plotted])
    rho = float(spearmanr(elements, distortion).statistic)

    mpl.rcParams.update(
        {
            "font.size": 16.5,
            "axes.labelsize": 18.75,
            "xtick.labelsize": 15.75,
            "ytick.labelsize": 15.75,
            "legend.fontsize": 12.3,
        }
    )
    panel_w = 5.9736 * 1.3
    panel_h = 3.663 * 1.3
    left = 1.70
    right = 3.35
    top = 0.28
    bottom = 1.28
    fig_w = left + panel_w + right
    fig_h = top + panel_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h])
    style = dict(s=28, edgecolor="white", linewidths=0.35, alpha=0.85, zorder=3)
    for prototype, label, marker, color in FIG3_FAMILIES:
        chosen = [row for row in plotted if row["prototype"] == prototype]
        if not chosen:
            continue
        ax.scatter(
            [int(row["n_elements"]) + horizontal_jitter(str(row["structure_key"])) for row in chosen],
            [float(row["mean_abs_pct"]) for row in chosen],
            label=label,
            marker=marker,
            facecolor=color,
            **style,
        )
    ax.set_xlim(int(elements.min()) - 0.7, int(elements.max()) + 0.7)
    ax.set_xticks(range(int(elements.min()), int(elements.max()) + 1))
    ax.set_ylim(0.0, FIG3_MAX_DISTORTION)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    _style_descriptor_panel(ax)
    ax.set_xlabel("Number of elements\non the high-entropy site")
    ax.set_ylabel("overall lattice distortion (%)")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=12.3,
        handlelength=1.15,
        labelspacing=0.30,
        borderaxespad=0.0,
        markerscale=2,
    )
    pdf = OUT_DIR / "figure_s2.pdf"
    png = OUT_DIR / "figure_s2.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    table_path = OUT_DIR / "figure_s2.csv"
    fields = ["structure_key", "prototype", "family", "n_elements", "mean_abs_pct"]
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "structure_key": row["structure_key"],
                    "prototype": row["prototype"],
                    "family": row["family"],
                    "n_elements": row["n_elements"],
                    "mean_abs_pct": f"{float(row['mean_abs_pct']):.6g}",
                }
            )
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Wrote {table_path}")
    print(
        f"n={len(plotted)} plotted  omitted distortion>{FIG3_MAX_DISTORTION:g}%: "
        f"{len(records) - len(plotted)}  "
        f"elements {elements.min()}–{elements.max()}  "
        f"Spearman elements ρ={rho:.3f}"
    )


def main() -> None:
    fig1_main()
    fig2_main()
    records = descriptor_records()
    write_descriptor_table(records)
    figure_3(records)
    figure_s2(records)
    fig4_main()
    figs1_main()


if __name__ == "__main__":
    main()
