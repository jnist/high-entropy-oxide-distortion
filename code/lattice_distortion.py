"""Measure lattice distortion from pairwise atomic distances.

The ideal supercell and the relaxed structure are compared at the same
lattice: fractional coordinates of the supercell are kept, and its lattice
matrix is replaced by the relaxed lattice. Distortion of one pair is the
absolute percentage change in that pair's distance. The structure metric is
the mean of those percentages for every unordered pair inside the RDF cutoff.

Pair search follows the RDF cutoff (6.0 A) and bin width (0.1 A). Periodic
images extend far enough to include every neighbor inside the cutoff, which
matters for cells shorter than 6 A.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

CUTOFF = 6.0
BIN_SIZE = 0.1
N_BINS = int(round(CUTOFF / BIN_SIZE))
BIN_EDGES = np.linspace(0.0, CUTOFF, N_BINS + 1)
ANIONS = {"O", "F", "N", "S", "Se", "Te", "Cl", "Br", "I", "C"}
PAIR_CLASSES = ("all", "ca", "cc", "aa")
CLASS_CODE = {"ca": 0, "cc": 1, "aa": 2}


def structure_key(path: Path) -> str:
    """Normalize filenames so relaxed/supercell pairs share the same key."""
    return re.sub(r"_relaxed$", "", path.stem)


def prototype_from_key(key: str) -> str:
    match = re.match(r"row\d+_(.+)", key)
    return match.group(1) if match else key


def read_poscar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lattice rows, species symbols, and fractional coordinates."""
    lines = path.read_text().splitlines()
    scale = float(lines[1])
    lattice = (
        np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
        * scale
    )
    species_names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    cursor = 7
    if lines[cursor].strip()[0] in "sS":
        cursor += 1
    coord_type = lines[cursor].strip()[0].lower()
    cursor += 1
    n_atoms = sum(counts)
    fracs = np.array(
        [[float(x) for x in lines[cursor + k].split()[:3]] for k in range(n_atoms)],
        dtype=float,
    )
    if coord_type == "c":
        fracs = np.linalg.solve(lattice.T, fracs.T).T
    symbols: list[str] = []
    for species, count in zip(species_names, counts, strict=True):
        symbols.extend([species] * count)
    return lattice, np.asarray(symbols), fracs


def plane_spacings(lattice: np.ndarray) -> np.ndarray:
    """Distance between adjacent lattice planes normal to each real-space axis."""
    volume = abs(float(np.linalg.det(lattice)))
    cross = np.stack(
        [
            np.cross(lattice[1], lattice[2]),
            np.cross(lattice[2], lattice[0]),
            np.cross(lattice[0], lattice[1]),
        ]
    )
    return volume / np.linalg.norm(cross, axis=1)


def periodic_images(lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """Integer images that can place a neighbor inside ``cutoff``.

    The range along each axis is ``ceil(cutoff / plane spacing)``, the same
    bound used to build a periodic neighbor list.
    """
    spacings = plane_spacings(lattice)
    n_img = np.ceil(cutoff / spacings - 1e-8).astype(int)
    n_img = np.maximum(n_img, 1)
    axes = [np.arange(-int(n), int(n) + 1) for n in n_img]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def canonical_self_images(images: np.ndarray) -> np.ndarray:
    """Keep one direction of each atom-to-its-own-image pair.

    ``(i, i, image)`` is the same pair as ``(i, i, -image)``. The image whose
    first nonzero component is positive is kept.
    """
    nonzero = images != 0
    has_component = nonzero.any(axis=1)
    first_index = np.argmax(nonzero, axis=1)
    first_value = images[np.arange(len(images)), first_index]
    return has_component & (first_value > 0)


def pair_records(
    lattice: np.ndarray,
    frac_ideal: np.ndarray,
    frac_relaxed: np.ndarray,
    is_anion: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal distances, absolute percent changes, and pair-class codes.

    Neighbors are chosen on the ideal coordinates placed on ``lattice``.
    The relaxed distance uses the same atom indices and the same image, so a
    neighbor is not reassigned after relaxation. Each unordered pair is
    counted once. Class codes are 0 cation-anion, 1 cation-cation, 2 anion-anion.
    """
    cart_ideal = frac_ideal @ lattice
    cart_relaxed = frac_relaxed @ lattice
    n_atoms = len(frac_ideal)
    images = periodic_images(lattice, cutoff)
    shifts = images @ lattice
    self_ok = canonical_self_images(images)
    ideal_images = cart_ideal[None, :, :] + shifts[:, None, :]
    relaxed_images = cart_relaxed[None, :, :] + shifts[:, None, :]
    neighbor_index = np.arange(n_atoms)

    ideal_parts: list[np.ndarray] = []
    percent_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []

    for start in range(0, n_atoms, 48):
        stop = min(n_atoms, start + 48)
        atom_index = np.arange(start, stop)
        dist_ideal = np.linalg.norm(
            ideal_images[None, :, :, :] - cart_ideal[start:stop, None, None, :],
            axis=-1,
        )
        keep_pair = (neighbor_index[None, None, :] > atom_index[:, None, None]) | (
            (neighbor_index[None, None, :] == atom_index[:, None, None])
            & self_ok[None, :, None]
        )
        mask = keep_pair & (dist_ideal > 1e-8) & (dist_ideal <= cutoff)
        if not np.any(mask):
            continue

        dist_relaxed = np.linalg.norm(
            relaxed_images[None, :, :, :] - cart_relaxed[start:stop, None, None, :],
            axis=-1,
        )
        chosen_ideal = dist_ideal[mask]
        chosen_relaxed = dist_relaxed[mask]
        chosen_i = np.broadcast_to(atom_index[:, None, None], dist_ideal.shape)[mask]
        chosen_j = np.broadcast_to(neighbor_index[None, None, :], dist_ideal.shape)[mask]
        anion_i = is_anion[chosen_i]
        anion_j = is_anion[chosen_j]
        pair_class = np.where(
            anion_i == anion_j,
            np.where(anion_i, CLASS_CODE["aa"], CLASS_CODE["cc"]),
            CLASS_CODE["ca"],
        )

        ideal_parts.append(chosen_ideal)
        percent_parts.append(100.0 * np.abs(chosen_relaxed - chosen_ideal) / chosen_ideal)
        class_parts.append(pair_class.astype(np.int8, copy=False))

    if not ideal_parts:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=np.int8)
    return (
        np.concatenate(ideal_parts),
        np.concatenate(percent_parts),
        np.concatenate(class_parts),
    )


def class_mean(percent: np.ndarray, pair_class: np.ndarray, code: int) -> tuple[float, int]:
    selected = percent[pair_class == code]
    if selected.size == 0:
        return float("nan"), 0
    return float(selected.mean()), int(selected.size)


def binned_means(
    distance: np.ndarray,
    percent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair counts and mean absolute percent in each RDF bin."""
    counts, _ = np.histogram(distance, bins=BIN_EDGES)
    weighted, _ = np.histogram(distance, bins=BIN_EDGES, weights=percent)
    means = np.full(counts.shape, np.nan, dtype=float)
    nonzero = counts > 0
    means[nonzero] = weighted[nonzero] / counts[nonzero]
    return counts.astype(int), means


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.8g}"


def bin_label(r_min: float, r_max: float) -> str:
    return f"[{r_min:.3f} - {r_max:.3f}]A"


def measure_pair(
    supercell_path: Path,
    relaxed_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lattice_ideal, symbols_ideal, frac_ideal = read_poscar(supercell_path)
    lattice_relaxed, symbols_relaxed, frac_relaxed = read_poscar(relaxed_path)
    if len(symbols_ideal) != len(symbols_relaxed) or not np.array_equal(
        symbols_ideal, symbols_relaxed
    ):
        raise ValueError(f"species mismatch: {supercell_path.name}")

    length_ideal = np.linalg.norm(lattice_ideal, axis=1)
    length_relaxed = np.linalg.norm(lattice_relaxed, axis=1)
    length_ratio = length_relaxed / length_ideal
    volume_ratio = abs(float(np.linalg.det(lattice_relaxed))) / abs(
        float(np.linalg.det(lattice_ideal))
    )

    is_anion = np.isin(symbols_ideal, list(ANIONS))
    distance, percent, pair_class = pair_records(
        lattice_relaxed, frac_ideal, frac_relaxed, is_anion, CUTOFF
    )
    if percent.size == 0:
        raise ValueError(f"no pairs within {CUTOFF} A: {supercell_path.name}")

    key = structure_key(supercell_path)
    mean_ca, n_ca = class_mean(percent, pair_class, CLASS_CODE["ca"])
    mean_cc, n_cc = class_mean(percent, pair_class, CLASS_CODE["cc"])
    mean_aa, n_aa = class_mean(percent, pair_class, CLASS_CODE["aa"])
    summary = {
        "structure_key": key,
        "prototype": prototype_from_key(key),
        "supercell_filename": supercell_path.name,
        "relaxed_filename": relaxed_path.name,
        "n_atoms": int(len(symbols_ideal)),
        "n_pairs": int(percent.size),
        "mean_abs_pct": float(percent.mean()),
        "median_abs_pct": float(np.median(percent)),
        "n_pairs_ca": n_ca,
        "mean_abs_pct_ca": mean_ca,
        "n_pairs_cc": n_cc,
        "mean_abs_pct_cc": mean_cc,
        "n_pairs_aa": n_aa,
        "mean_abs_pct_aa": mean_aa,
        "lattice_scale_a": float(length_ratio[0]),
        "lattice_scale_b": float(length_ratio[1]),
        "lattice_scale_c": float(length_ratio[2]),
        "volume_ratio": float(volume_ratio),
    }

    bin_rows: list[dict[str, object]] = []
    selections = {"all": np.ones(percent.size, dtype=bool)}
    for name, code in CLASS_CODE.items():
        selections[name] = pair_class == code
    for pair_name in PAIR_CLASSES:
        selected = selections[pair_name]
        counts, means = binned_means(distance[selected], percent[selected])
        for index, (count, mean) in enumerate(zip(counts, means, strict=True)):
            r_min = float(BIN_EDGES[index])
            r_max = float(BIN_EDGES[index + 1])
            bin_rows.append(
                {
                    "structure_key": key,
                    "pair_class": pair_name,
                    "bin_label": bin_label(r_min, r_max),
                    "r_min_A": r_min,
                    "r_max_A": r_max,
                    "r_center_A": 0.5 * (r_min + r_max),
                    "n_pairs": int(count),
                    "mean_abs_pct": mean,
                }
            )
    return summary, bin_rows


SUMMARY_FIELDS = [
    "structure_key",
    "prototype",
    "supercell_filename",
    "relaxed_filename",
    "n_atoms",
    "n_pairs",
    "mean_abs_pct",
    "median_abs_pct",
    "n_pairs_ca",
    "mean_abs_pct_ca",
    "n_pairs_cc",
    "mean_abs_pct_cc",
    "n_pairs_aa",
    "mean_abs_pct_aa",
    "lattice_scale_a",
    "lattice_scale_b",
    "lattice_scale_c",
    "volume_ratio",
]
FLOAT_SUMMARY_FIELDS = {
    "mean_abs_pct",
    "median_abs_pct",
    "mean_abs_pct_ca",
    "mean_abs_pct_cc",
    "mean_abs_pct_aa",
    "lattice_scale_a",
    "lattice_scale_b",
    "lattice_scale_c",
    "volume_ratio",
}
BIN_FIELDS = [
    "structure_key",
    "pair_class",
    "bin_label",
    "r_min_A",
    "r_max_A",
    "r_center_A",
    "n_pairs",
    "mean_abs_pct",
]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in FLOAT_SUMMARY_FIELDS:
                output[field] = fmt(float(output[field]))
            writer.writerow(output)


def write_bins(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BIN_FIELDS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in ("r_min_A", "r_max_A", "r_center_A"):
                output[field] = f"{float(output[field]):.3f}"
            output["mean_abs_pct"] = fmt(float(output["mean_abs_pct"]))
            writer.writerow(output)


def paired_paths(
    supercells_dir: Path, relaxed_dir: Path
) -> list[tuple[str, Path, Path]]:
    supercells = {structure_key(path): path for path in supercells_dir.glob("*.vasp")}
    relaxed = {structure_key(path): path for path in relaxed_dir.glob("*.vasp")}
    common = sorted(set(supercells) & set(relaxed))
    missing_super = sorted(set(relaxed) - set(supercells))
    missing_relaxed = sorted(set(supercells) - set(relaxed))
    if missing_super or missing_relaxed:
        print(
            "Warning: unmatched keys "
            f"(relaxed only: {len(missing_super)}, supercell only: {len(missing_relaxed)})",
            file=sys.stderr,
        )
    return [(key, supercells[key], relaxed[key]) for key in common]


def main() -> int:
    release = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Absolute percentage lattice distortion from pairwise distances."
    )
    parser.add_argument(
        "--supercells",
        type=Path,
        default=release / "data" / "structures" / "supercells",
        help="Directory of ideal supercell POSCAR files.",
    )
    parser.add_argument(
        "--relaxed",
        type=Path,
        default=release / "data" / "structures" / "relaxed" / "equflashv2",
        help="Directory of relaxed POSCAR files for one potential.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/distortion/recomputed).",
    )
    args = parser.parse_args()

    supercells_dir = args.supercells.resolve()
    relaxed_dir = args.relaxed.resolve()
    output_dir = (
        args.output_dir or release / "data" / "distortion" / "recomputed"
    ).resolve()
    for directory in (relaxed_dir, supercells_dir):
        if not directory.is_dir():
            print(f"Missing directory: {directory}", file=sys.stderr)
            return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = paired_paths(supercells_dir, relaxed_dir)
    print(f"Computing absolute percentage distortion up to {CUTOFF} A")
    print(f"Pairs: {len(pairs)}")
    print(f"Output: {output_dir}")

    summaries: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    errors: list[str] = []
    start = time.time()
    for index, (_key, supercell_path, relaxed_path) in enumerate(pairs, start=1):
        try:
            summary, bins = measure_pair(supercell_path, relaxed_path)
        except Exception as exc:  # noqa: BLE001 - collect and continue the batch
            errors.append(f"{supercell_path.name}: {exc}")
            continue
        summaries.append(summary)
        bin_rows.extend(bins)
        if index % 50 == 0 or index == len(pairs):
            print(f"  {index}/{len(pairs)} pairs handled")

    summary_path = output_dir / "lattice_distortion.csv"
    bins_path = output_dir / "lattice_distortion_bins.csv"
    write_summary(summary_path, summaries)
    write_bins(bins_path, bin_rows)

    elapsed = time.time() - start
    print(
        f"Done in {elapsed:.1f}s: {len(summaries)} structures written, "
        f"{len(errors)} failed"
    )
    print(f"Summary: {summary_path}")
    print(f"Bins: {bins_path}")
    if summaries:
        values = np.array([float(row["mean_abs_pct"]) for row in summaries])
        print(
            "Mean absolute percent: "
            f"min={values.min():.4f}, "
            f"median={np.median(values):.4f}, "
            f"mean={values.mean():.4f}, "
            f"max={values.max():.4f}"
        )
    if errors:
        error_path = output_dir / "lattice_distortion_errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Errors logged to: {error_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
