"""Atomic-size mismatch of the EquFlash high-entropy oxides.

Equation 2 of Song et al., Phys. Rev. Materials 1, 023404 (2017), Zhang's
atomic-size mismatch, evaluated on the high-entropy cation site:

    delta = 100 * sqrt( sum_i c_i (1 - r_i / r_bar)^2 )
    r_bar = sum_i c_i r_i

c_i is the atomic fraction of element i on that site. r_i is the Shannon
ionic radius from pymatgen at the site's oxygen coordination. Vacancies (X)
are left out of the sum.

The occupation is the sorted_json_formula column of data/compositions.csv.
A relaxed file row619_rock-salt_relaxed.vasp is composition row 619.
Garnet, NASICON, and rows with no oxygen are excluded. Coordination is
measured on the ordered prototype cell: anion neighbours are kept in one
shell until a gap larger than 0.38 Å. That gap leaves the Ruddlesden-Popper
rare-earth shell at 8 and splits Gd2Si2O7 into coordinations 7 and 8.
KCoO2 uses the NaCoO2 conventional cell.

Run from the release directory::

    python code/atomic_size_mismatch.py
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter
from functools import reduce
from itertools import product
from math import gcd
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element, Species

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "compositions.csv"
RELAXED = ROOT / "data" / "structures" / "relaxed" / "equflashv2"
PROTOTYPES = ROOT / "data" / "structures" / "prototypes"
OUT_DIR = ROOT / "data" / "descriptors"

# First shell ends at the first anion-anion gap wider than this. 0.38 Å sits
# between the 0.36 Å split inside the RP rare-earth shell and the 0.38 Å gap
# that separates the seventh oxygen from the rest in Gd2Si2O7.
SHELL_GAP = 0.38  # Å
MIN_DISTANCE = 0.4  # Å
# An ICSD-occurrence penalty per unit of charge residual. 1000 means a
# residual of 0.1 costs as much as 100 tabulated occurrences, so a rare
# oxidation state is not chosen just to absorb a rounding error, while a
# residual of 0.2 still outweighs Cu+ over Cu2+ (52 occurrences).
CHARGE_PENALTY = 1000.0
MAX_OXIDATION_COMBINATIONS = 400_000
EXCLUDE_PROTOTYPES = {"garnet", "NASICON"}
ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N"}
ROMAN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    14: "XIV",
}
ROMAN_TO_INT = {label: number for number, label in ROMAN.items()}

RESULT_FIELDS = [
    "row",
    "structure",
    "prototype_name",
    "formula",
    "he_site",
    "he_fractions",
    "n_elements",
    "delta",
    "delta_percent",
    "r_bar",
    "oxidation_states",
    "coordination",
    "shannon_radii",
    "note",
]
SKIP_FIELDS = ["row", "structure", "prototype_name", "formula", "reason"]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def prototype_path(prototype_name: str) -> Path:
    """Ordered cell for a dataset prototype. KCoO2 uses the NaCoO2 cell."""
    wanted = "NaCoO2" if prototype_name == "KCoO2" else prototype_name
    key = normalize_name(wanted)
    matches = [
        path
        for path in PROTOTYPES.glob("*.vasp")
        if normalize_name(path.stem.split("_", 1)[-1]) == key
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"no unique prototype cell for {prototype_name}: {matches}")
    return matches[0]


def read_poscar(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    cursor = 7
    if lines[cursor].strip()[0] in "sS":
        cursor += 1
    cursor += 1
    n_atoms = sum(counts)
    frac = np.array(
        [[float(x) for x in lines[cursor + k].split()[:3]] for k in range(n_atoms)],
        dtype=float,
    )
    symbols: list[str] = []
    for name, count in zip(names, counts, strict=True):
        symbols.extend([name] * count)
    return lattice, symbols, frac


def anion_distances(lattice: np.ndarray, symbols: list[str], frac: np.ndarray, index: int) -> np.ndarray:
    cart = frac @ lattice
    shifts = np.array(
        [
            a * lattice[0] + b * lattice[1] + c * lattice[2]
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
        ]
    )
    images = cart[None, :, :] + shifts[:, None, :]
    distances = np.linalg.norm(images - cart[index], axis=-1).ravel()
    anion = np.array([symbol in ANIONS for symbol in symbols])
    mask = np.tile(anion, len(shifts))
    kept = distances[(mask) & (distances > MIN_DISTANCE)]
    return np.sort(kept)


def shell_size(distances: np.ndarray) -> int:
    if distances.size == 0:
        return 0
    count = 1
    for previous, current in zip(distances, distances[1:], strict=False):
        if current - previous > SHELL_GAP:
            break
        count += 1
    return count


def cation_coordinations(path: Path) -> list[tuple[str, Counter[int]]]:
    """Cation species in POSCAR order, each with a count of first-shell sizes."""
    lattice, symbols, frac = read_poscar(path)
    grouped: dict[str, Counter[int]] = {}
    order: list[str] = []
    for index, symbol in enumerate(symbols):
        if symbol in ANIONS:
            continue
        if symbol not in grouped:
            order.append(symbol)
            grouped[symbol] = Counter()
        grouped[symbol][shell_size(anion_distances(lattice, symbols, frac, index))] += 1
    return [(symbol, grouped[symbol]) for symbol in order]


def prototype_multiplicities(path: Path) -> dict[str, float]:
    """Reduced A, B, C, ... multiplicities in POSCAR species order."""
    _lattice, symbols, _frac = read_poscar(path)
    order: list[str] = []
    counts: Counter[str] = Counter()
    for symbol in symbols:
        if symbol not in counts:
            order.append(symbol)
        counts[symbol] += 1
    divisor = reduce(gcd, counts.values())
    letters = [chr(ord("A") + index) for index in range(len(order))]
    return {letter: counts[symbol] / divisor for letter, symbol in zip(letters, order, strict=True)}


def parse_prototype_formula(value: str) -> dict[str, float] | None:
    if value is None or value == "" or value.lower() == "nan":
        return None
    parsed = ast.literal_eval(value)
    return {str(key): float(amount) for key, amount in parsed.items()}


def contains_oxygen(formula: dict[str, dict[str, float]]) -> bool:
    return any("O" in occupants for occupants in formula.values())


def is_anion_site(occupants: dict[str, float]) -> bool:
    """Oxygen, oxygen plus vacancy, or a site with no metallic element."""
    elements = [element for element in occupants if element != "X" and occupants[element] > 0]
    if not elements:
        return True
    if "O" in occupants and occupants["O"] > 0:
        return True
    return all(element in ANIONS for element in elements)


def real_fractions(occupants: dict[str, float]) -> dict[str, float]:
    """Element fractions on a site, vacancies removed and renormalized."""
    weights = {
        element: float(amount)
        for element, amount in occupants.items()
        if element != "X" and float(amount) > 0
    }
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {element: amount / total for element, amount in weights.items()}


def mixing_entropy(fractions: dict[str, float]) -> float:
    if len(fractions) <= 1:
        return 0.0
    return float(-sum(fraction * math.log(fraction) for fraction in fractions.values()))


def high_entropy_site(
    formula: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float]]:
    best: tuple[float, int, str] | None = None
    chosen: dict[str, float] = {}
    for letter, occupants in formula.items():
        if is_anion_site(occupants):
            continue
        fractions = real_fractions(occupants)
        if not fractions:
            continue
        rank = (mixing_entropy(fractions), len(fractions), letter)
        if best is None or rank > best:
            best = rank
            chosen = fractions
    if best is None:
        raise ValueError("no cation site")
    return best[2], chosen


def site_coordinations(
    formula: dict[str, dict[str, float]],
    cations: list[tuple[str, Counter[int]]],
) -> dict[str, Counter[int]]:
    """Map each formula cation letter onto the prototype coordination counts."""
    cation_sites = [letter for letter, occupants in formula.items() if not is_anion_site(occupants)]
    assigned: dict[str, Counter[int]] = {}
    if len(cation_sites) == len(cations):
        for letter, (_symbol, counts) in zip(cation_sites, cations, strict=True):
            assigned[letter] = counts
        return assigned
    if len(cations) == 1:
        # One crystallographic cation site, split across several formula letters.
        for letter in cation_sites:
            assigned[letter] = cations[0][1]
        return assigned
    if len(cation_sites) == 1:
        merged: Counter[int] = Counter()
        for _symbol, counts in cations:
            merged.update(counts)
        assigned[cation_sites[0]] = merged
        return assigned
    raise ValueError(
        f"cannot map cation sites {cation_sites} onto prototype species "
        f"{[symbol for symbol, _counts in cations]}"
    )


def composition_multiplicities(
    formula: dict[str, dict[str, float]],
    prototype_formula: dict[str, float] | None,
    cell_multiplicities: dict[str, float],
    cations: list[tuple[str, Counter[int]]],
) -> dict[str, float]:
    """Atoms of each formula letter in the reduced prototype."""
    if prototype_formula is not None and set(prototype_formula) == set(formula):
        return dict(prototype_formula)

    cation_sites = [letter for letter, occupants in formula.items() if not is_anion_site(occupants)]
    anion_sites = [letter for letter in formula if letter not in cation_sites]
    if len(cations) == 1 and len(cation_sites) > 1 and prototype_formula is not None:
        # Rock salt recorded as two cation letters. Each letter is a full
        # cation site, and the anion count keeps the prototype cation:anion ratio.
        proto_cation = next(amount for letter, amount in cell_multiplicities.items() if letter == "A")
        proto_anion = sum(cell_multiplicities.values()) - proto_cation
        multiplicities = {letter: proto_cation for letter in cation_sites}
        anion_total = proto_cation * len(cation_sites) * proto_anion / proto_cation
        share = anion_total / max(len(anion_sites), 1)
        for letter in anion_sites:
            multiplicities[letter] = share
        return multiplicities

    if len(cation_sites) == len(cations):
        # Letters follow the prototype species order. Used when the dataset
        # has no prototype_formula, as for LaTaO4.
        letters = list(cell_multiplicities)
        return {letter: cell_multiplicities[source] for letter, source in zip(cation_sites + anion_sites, letters, strict=False)}
    raise ValueError("no site multiplicities")


def formula_amounts(
    formula: dict[str, dict[str, float]], multiplicities: dict[str, float]
) -> dict[str, float]:
    amounts: Counter[str] = Counter()
    for letter, occupants in formula.items():
        for element, fraction in occupants.items():
            if element == "X":
                continue
            amounts[element] += float(fraction) * multiplicities[letter]
    return {element: amount for element, amount in amounts.items() if amount > 1e-8}


def oxidation_probabilities() -> dict[Species, float]:
    """ICSD occurrence weights used by Composition.oxi_state_guesses."""
    if Composition.oxi_prob is None:
        Composition("MgO").oxi_state_guesses()
    return Composition.oxi_prob or {}


def candidate_states(element: str, probabilities: dict[Species, float]) -> list[int]:
    states = Element(element).icsd_oxidation_states or Element(element).common_oxidation_states
    # A state with no Shannon radius cannot be used for the mismatch. La2+ and
    # Ni+ are examples: they improve a formal charge sum and then fail later.
    usable = [
        int(state)
        for state in states
        if available_coordinations(shannon_table(element), int(state))
    ]
    ordered = usable or [int(state) for state in states]
    ordered.sort(key=lambda state: probabilities.get(Species(element, state), 0.0), reverse=True)
    return ordered or [0]


def trim_states(
    states: dict[str, list[int]], probabilities: dict[Species, float]
) -> dict[str, list[int]]:
    """Keep the product of state lists small enough to enumerate."""
    trimmed = {element: list(options) for element, options in states.items()}

    def width() -> int:
        total = 1
        for options in trimmed.values():
            total *= max(len(options), 1)
        return total

    while width() > MAX_OXIDATION_COMBINATIONS:
        element = max(trimmed, key=lambda name: len(trimmed[name]))
        if len(trimmed[element]) <= 1:
            break
        trimmed[element].pop()  # least probable, lists are sorted high to low
    return trimmed


def oxidation_guesses(amounts: dict[str, float]) -> list[tuple[dict[str, int], float]]:
    """Most probable oxidation state of each element, and its charge residual.

    One state per element, which is what Composition.oxi_state_guesses returns
    when the average on that element is an integer. States and weights are the
    ICSD oxidation states pymatgen uses. The rank is the ICSD weight minus
    CHARGE_PENALTY times the absolute charge residual, so a published formula
    that is only approximately neutral keeps the common oxide states.
    """
    probabilities = oxidation_probabilities()
    elements = [element for element, amount in amounts.items() if amount > 0]
    states = trim_states(
        {element: candidate_states(element, probabilities) for element in elements},
        probabilities,
    )
    ranked: list[tuple[float, float, dict[str, int]]] = []
    for combo in product(*(states[element] for element in elements)):
        assignment = dict(zip(elements, combo, strict=True))
        charge = abs(
            sum(amounts[element] * oxidation for element, oxidation in assignment.items())
        )
        score = sum(
            probabilities.get(Species(element, oxidation), 0.0)
            for element, oxidation in assignment.items()
        )
        objective = score - CHARGE_PENALTY * charge
        ranked.append((objective, charge, assignment))
    ranked.sort(key=lambda item: -item[0])
    # The first assignment that has a Shannon radius is used. Keep a short list.
    return [(assignment, charge) for _objective, charge, assignment in ranked[:30]]


def shannon_table(element: str) -> dict:
    data = Element(element).data.get("Shannon radii")
    if not isinstance(data, dict):
        return {}
    return data


def available_coordinations(table: dict, oxidation: int) -> dict[int, str]:
    entry = table.get(str(oxidation), {})
    found: dict[int, str] = {}
    for label in entry:
        if label in ROMAN_TO_INT:
            found[ROMAN_TO_INT[label]] = label
    return found


def pick_spin(spins: dict) -> str:
    if "High Spin" in spins:
        return "High Spin"
    if "" in spins:
        return ""
    if "Low Spin" in spins:
        return "Low Spin"
    return next(iter(spins))


def radius_at(
    element: str, oxidation: int, coordination: int
) -> tuple[float, int, str, bool]:
    """Ionic radius, coordination actually used, spin, and whether CN was substituted."""
    table = shannon_table(element)
    available = available_coordinations(table, oxidation)
    if not available:
        raise KeyError(f"no Shannon radius for {element}{oxidation:+}")
    substituted = coordination not in available
    used = coordination
    if substituted:
        used = min(available, key=lambda number: (abs(number - coordination), number))
    spins = table[str(oxidation)][available[used]]
    spin = pick_spin(spins)
    return float(spins[spin]["ionic_radius"]), used, spin, substituted


def weighted_radius(
    element: str, oxidation: int, counts: Counter[int]
) -> tuple[float, list[str]]:
    notes: list[str] = []
    total = 0.0
    weight = 0
    for coordination, count in sorted(counts.items()):
        radius, used, _spin, substituted = radius_at(element, oxidation, coordination)
        if substituted:
            notes.append(f"{element}{oxidation:+} CN {coordination}->{used}")
        total += radius * count
        weight += count
    if weight == 0:
        raise KeyError(f"no coordination count for {element}")
    return total / weight, notes


def format_coordination(counts: Counter[int]) -> str:
    if len(counts) == 1:
        return str(next(iter(counts)))
    return ";".join(f"{coordination}:{count}" for coordination, count in sorted(counts.items()))


def format_map(values: dict[str, object], digits: int | None = None) -> str:
    parts: list[str] = []
    for key, value in values.items():
        if digits is None:
            parts.append(f"{key}:{value}")
        else:
            parts.append(f"{key}:{float(value):.{digits}f}")
    return ";".join(parts)


def size_mismatch(fractions: dict[str, float], radii: dict[str, float]) -> tuple[float, float, float]:
    r_bar = sum(fractions[element] * radii[element] for element in fractions)
    if r_bar <= 0:
        raise ValueError("mean radius is not positive")
    variance = sum(
        fractions[element] * (1.0 - radii[element] / r_bar) ** 2 for element in fractions
    )
    delta = math.sqrt(variance)
    return delta, 100.0 * delta, r_bar


def usable_oxidation(guess: dict[str, float], elements: list[str]) -> dict[str, int] | None:
    oxidation: dict[str, int] = {}
    for element in elements:
        if element not in guess:
            return None
        value = float(guess[element])
        rounded = int(round(value))
        if abs(value - rounded) > 1e-3:
            return None
        if not available_coordinations(shannon_table(element), rounded):
            return None
        oxidation[element] = rounded
    return oxidation


def evaluate_row(
    formula: dict[str, dict[str, float]],
    prototype_name: str,
    prototype_formula: dict[str, float] | None,
    coordinations: dict[str, list[tuple[str, Counter[int]]]],
    cell_multiplicities: dict[str, dict[str, float]],
) -> dict[str, object]:
    letter, fractions = high_entropy_site(formula)
    counts = site_coordinations(formula, coordinations[prototype_name])[letter]
    multiplicities = composition_multiplicities(
        formula,
        prototype_formula,
        cell_multiplicities[prototype_name],
        coordinations[prototype_name],
    )
    guesses = oxidation_guesses(formula_amounts(formula, multiplicities))
    elements = list(fractions)
    chosen: dict[str, int] | None = None
    guess_index = 0
    residual = math.inf
    for guess_index, (guess, residual) in enumerate(guesses):
        chosen = usable_oxidation(guess, elements)
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError("no charge-balanced oxidation state with a Shannon radius")

    radii: dict[str, float] = {}
    notes: list[str] = []
    if guess_index > 0:
        notes.append(f"oxidation guess {guess_index + 1}")
    if residual > 0.02:
        notes.append(f"charge residual {residual:.3f}")
    for element in elements:
        radius, radius_notes = weighted_radius(element, chosen[element], counts)
        radii[element] = radius
        notes.extend(radius_notes)
    delta, delta_percent, r_bar = size_mismatch(fractions, radii)
    return {
        "he_site": letter,
        "he_fractions": format_map(fractions, 4),
        "n_elements": len(fractions),
        "delta": f"{delta:.6f}",
        "delta_percent": f"{delta_percent:.4f}",
        "r_bar": f"{r_bar:.4f}",
        "oxidation_states": format_map({element: f"{chosen[element]:+d}" for element in elements}),
        "coordination": format_coordination(counts),
        "shannon_radii": format_map(radii, 4),
        "note": "; ".join(notes),
    }


def load_dataset() -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with DATASET.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows[int(record["row"])] = record
    return rows


def relaxed_rows() -> dict[int, str]:
    rows: dict[int, str] = {}
    for path in RELAXED.glob("*.vasp"):
        match = re.match(r"row(\d+)_", path.name)
        if match is None:
            continue
        rows[int(match.group(1))] = path.stem
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    dataset = load_dataset()
    present = relaxed_rows()
    names = sorted(
        {
            dataset[row]["prototype_name"]
            for row in present
            if row in dataset
            and dataset[row]["prototype_name"] not in EXCLUDE_PROTOTYPES
            and contains_oxygen(json.loads(dataset[row]["sorted_json_formula"]))
        }
    )
    coordinations = {name: cation_coordinations(prototype_path(name)) for name in names}
    cell_multiplicities = {
        name: prototype_multiplicities(prototype_path(name)) for name in names
    }
    print("Prototype coordinations")
    for name in names:
        rendered = ", ".join(
            f"{symbol} {format_coordination(counts)}" for symbol, counts in coordinations[name]
        )
        print(f"  {name}: {rendered}", flush=True)

    results: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    n_oxides = 0
    for row in sorted(present):
        if row not in dataset:
            skipped.append(
                {
                    "row": row,
                    "structure": present[row],
                    "prototype_name": "",
                    "formula": "",
                    "reason": "row is past the end of dataset.csv",
                }
            )
            continue
        record = dataset[row]
        prototype_name = record["prototype_name"]
        formula_text = record["sorted_json_formula"]
        formula = json.loads(formula_text)
        if prototype_name in EXCLUDE_PROTOTYPES or not contains_oxygen(formula):
            continue
        n_oxides += 1
        if n_oxides % 50 == 0:
            print(f"  {n_oxides} oxides", flush=True)
        try:
            measured = evaluate_row(
                formula,
                prototype_name,
                parse_prototype_formula(record.get("prototype_formula", "")),
                coordinations,
                cell_multiplicities,
            )
        except (ValueError, KeyError, ZeroDivisionError) as error:
            skipped.append(
                {
                    "row": row,
                    "structure": present[row],
                    "prototype_name": prototype_name,
                    "formula": formula_text,
                    "reason": str(error),
                }
            )
            continue
        results.append(
            {
                "row": row,
                "structure": present[row],
                "prototype_name": prototype_name,
                "formula": formula_text,
                **measured,
            }
        )

    results.sort(key=lambda item: int(item["row"]))
    skipped.sort(key=lambda item: int(item["row"]))
    write_csv(OUT_DIR / "atomic_size_mismatch.csv", RESULT_FIELDS, results)
    write_csv(OUT_DIR / "skipped_oxides.csv", SKIP_FIELDS, skipped)
    print(f"Wrote {len(results)} oxides and {len(skipped)} skipped rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
