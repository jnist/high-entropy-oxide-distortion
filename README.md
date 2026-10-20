# Lattice distortion of high-entropy oxides

Data and code for the lattice-distortion analysis of 427 high-entropy oxide compositions in 14 prototype families. Each composition was relaxed with five universal machine-learning interatomic potentials. Lattice distortion is the mean absolute percentage change of every pairwise distance inside 6 Å between the ideal supercell and the relaxed structure, compared in the relaxed cell so that a uniform change of lattice parameter does not contribute.

## Layout

```text
data/
  compositions.csv            427 site-resolved formulas, with source DOIs
  distortion/                 pairwise-distance summaries for each potential
  descriptors/                Shannon-radius atomic size mismatch
  structures/
    prototypes/               ordered conventional cells
    supercells/               ideal POSCARs, about 400 atoms
    relaxed/                  relaxed POSCARs for five potentials
  relaxation_summary.csv      convergence record, without local file paths
code/
  lattice_distortion.py       equation for the pairwise-distance metric
  atomic_size_mismatch.py     size mismatch on the high-entropy cation site
  plot_figures.py             manuscript figures
figures/vesta/                renderings used in Figure 1
```

The 14 families are rock salt (53), fluorite (30), cubic perovskite (40), Ruddlesden–Popper perovskite (6), ternary spinel (11), KCoO2 (5), NaCoO2 (51), binary spinel (27), pyrochlore (94), tungsten bronze (5), orthorhombic perovskite (51), Gd2SiO5 (22), LaTaO4 (11) and Gd2Si2O7 (21). One LaTaO4 entry is an isostructural phosphate. `row` in `compositions.csv` is the index of that composition in the source table, and it matches the `rowNNN` prefix of every structure file.

KCoO2 uses the NaCoO2 conventional cell (`mp-867515_NaCoO2.vasp`), so `structures/prototypes/` contains 13 files for the 14 families.

## Reproduce the figures

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python code/plot_figures.py
```

PDF, PNG, and CSV outputs are written to `figures/`. Figure 1 assembles the VESTA renders. Figure 2 reads the five distortion summaries. Figures 3 and S2 recompute the ideal mixing entropy from the supercells and read the atomic-size mismatch table. Figure 4 recomputes cation–cation and oxygen–oxygen shells from the EquFlash v2 structures. Figure S1 is a schematic and does not read the structure files.

Compositions with an overall distortion above 10% are omitted from Figures 2b, 3, 4, and S2. They remain in the distortion tables.

## Recompute the distortion metric

```bash
python code/lattice_distortion.py \
  --supercells data/structures/supercells \
  --relaxed data/structures/relaxed/equflashv2 \
  --output-dir data/distortion/recomputed
```

Replace `equflashv2` with `esen`, `mace`, `mattersim`, or `dpa` to repeat the metric for another potential. The command writes `lattice_distortion.csv` and `lattice_distortion_bins.csv`. The published summaries are the CSV files already in `data/distortion/`. `model_agreement.csv` records, for each composition, which potential lies closest to the five-potential average.

## Atomic size mismatch

```bash
python code/atomic_size_mismatch.py
```

This rewrites `data/descriptors/atomic_size_mismatch.csv`. The mismatch is Zhang's atomic-size parameter on the high-entropy cation site, using Shannon ionic radii at the oxygen coordination measured on the ordered prototype. It needs pymatgen.

## Relaxations

Positions and the cell were relaxed together with the FIRE optimizer and the Frechet cell filter to a maximum residual force of 0.02 eV/Å, with at most 500 steps. `relaxation_summary.csv` records the checkpoint, step count, final energy, final maximum force, and whether that force target was met. The relaxed POSCARs are the structures used for the distortion metric. The relaxation itself depends on the published checkpoints of EquFlash v2, eSEN-30M-OAM, DPA-3.1-3M-FT, MatterSim v1 5M, and MACE-OMAT-0, which are not redistributed here.

## License

Code in `code/` is MIT. Data in `data/` and the VESTA renders in `figures/vesta/` are CC BY 4.0. See [LICENSE](LICENSE).
