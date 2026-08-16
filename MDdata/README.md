# Nhp6A–DNA metadynamics — simulation and analysis package

Input structures, force-field files, simulation inputs and the analysis/plotting code behind the
molecular dynamics results reported in the accompanying paper on the yeast HMG-box protein Nhp6A
and its complexes with DNA.

This is a **methods and code package**: everything needed to set up, launch and analyse the
simulations is here. The raw trajectories, metadynamics `HILLS` and collective-variable time
series are too large to distribute and are not included (see
[Data not included](#data-not-included)).

## Systems

All complexes use a 93-residue Nhp6A monomer (HMG box plus its disordered N-terminal region) and
a 15-bp B-DNA duplex.

| Name | Composition | Sampling |
|---|---|---|
| `DNA` | free 15-bp duplex | 1D well-tempered metadynamics on the DNA end-to-end distance, 6 walkers |
| `1Nhp6A` | 1:1 Nhp6A–DNA, wild type | 2D well-tempered metadynamics on (IDR–DNA contacts, end-to-end distance), 6 walkers |
| `1S26D`, `1T63D`, `1TM` | 1:1 complexes of the phosphomimetic variants S26D, T63D and the triple mutant S26D/S41D/T63D | as `1Nhp6A` |
| `1Nhp6A_apo_large` | Nhp6A alone (no DNA), enlarged box | 2D well-tempered metadynamics on (fraction of native core contacts, IDR–core contacts), 6 walkers |
| `2Nhp6A` | 2:1 Nhp6A–DNA, wild type | unbiased equilibrium MD, 3 replicas |

Unbiased equilibrium MD (3 replicas per system) was additionally run for the free DNA duplex and
for the 1:1 and 2:1 wild-type complexes, with PLUMED used in monitor-only mode to record the same
collective variables without applying any bias.

## Simulation protocol

| | |
|---|---|
| Engine | GROMACS 2024.2, patched with PLUMED 2.9 |
| Force field | AMBER ff14SB (protein) + parmbsc1 (DNA), supplied as `structures_topology/amber14sb_parmbsc1.ff` |
| Water / ions | TIP3P, 0.15 M KCl, truncated-octahedron box |
| Temperature | 298 K |
| Enhanced sampling | Well-tempered metadynamics, multiple walkers (6), `PACE=500` |
| Bias (complexes and free DNA) | height 2.0 kJ/mol, bias factor 5, σ = 3.0 (contacts) / 0.25 nm (end-to-end distance) |
| Bias (apo protein) | height 1.2 kJ/mol, bias factor 10, σ = 0.025 (contact fraction) / 1.5 (contacts) |
| Restraints on the biased runs | soft `LOWER_WALLS` on the DNA inter-strand contact map (duplex integrity, inactive above Q = 0.9) for the DNA-containing systems; `UPPER_WALLS` on core RMSD (0.5 nm) for the apo protein |

Starting structures were built from AlphaFold 3 predictions of the complexes; the raw predictions
and the cleaned-up versions used for system preparation are in `structures_topology/AF_Nhp6A/`.

## Contents

```
structures_topology/    Starting structures and GROMACS topologies, one directory per system,
                        plus AF_Nhp6A/ (AlphaFold models) and amber14sb_parmbsc1.ff/ (force field)
setup/                  System-preparation scripts (pdb2gmx → solvate → genion → EM → NVT/NPT),
                        mdp/ run-parameter files, PLUMED input generation, DNA contact-map
                        definition, PLUMED atom-group extraction
plumed/                 PLUMED input files as run: biased metadynamics for each system and the
                        monitor-only (unbiased) inputs used for the equilibrium simulations
analysis/scripts/       Convergence assessment, reweighting, free-energy/PMF construction,
                        structure selection and figure generation
analysis/notebooks/     Notebooks that assemble the published figure panels from those scripts
structures_states/      Representative structures of the free-energy basins shown in the figures
```

### `structures_topology/`

Per system: the solvated starting structure (`*.pdb` / `*.gro`), the GROMACS topology
(`*_ions.top`) and its `#include`d chain topologies (`*.itp`, with `*_pr.itp` position restraints),
the PLUMED `CONTACTMAP` definition of DNA duplex integrity (`*_cmap.dat`, with the atom-pair
listing in `*_cmap.txt`), and, for the apo runs, the Cα reference structure used for the core
RMSD (`ref_core_ca_*.pdb`). All topologies include only the bundled force field, so no external
force-field installation is required.

`1TM/make_1TM_mutant.py` and `make_1TM_mutant.pml` build the triple-mutant structure from the
wild-type model.

### `setup/`

- `setup_amber_*.sh` — build each system from the AlphaFold model: fix DNA atom naming, run
  `pdb2gmx` with the bundled force field, define the box, solvate and neutralise to 0.15 M KCl.
- `setup_*.sh` — energy minimisation, NVT/NPT equilibration and production `.tpr` generation for
  the complexes and for the apo (protein-only) boxes.
- `mdp/` — GROMACS run-parameter files: `em.mdp` (minimisation), `nvt.mdp`, `npt.mdp`
  (equilibration), `md.mdp` (production), plus the position-restrained and equilibration variants
  `md_posre.mdp` and `md_equil.mdp`.
- `gen_plumed_inputs.py`, `gen_plumed_inputs_binary.py` — generate the PLUMED input for a system
  from its structure file (collective-variable atom groups, walls, walker count).
- `contact_map_setup.ipynb` — identifies the DNA inter-strand native contacts and writes the
  PLUMED `CONTACTMAP` block used as the duplex-integrity collective variable.
- `get_group_indices.py` — prints the PLUMED `GROUP` definitions (IDR Cα/Cβ atoms, DNA backbone
  P and C1′ atoms) for a given structure file.

The scripts are the versions used to build the systems and are written to run from a project root
holding `data/mdp/` and a `data/<system>/` output directory; adjust the `source .../GMXRC.bash`
line and those paths to your own installation and layout.

### `plumed/`

| File | Use |
|---|---|
| `plumed_meta_2D_1Nhp6A_wall.dat` | WT 1:1 complex, 2D metadynamics |
| `plumed_meta_2D_1S26D_wall.dat`, `..._1T63D_wall.dat`, `..._1TM_wall.dat` | mutant 1:1 complexes |
| `plumed_meta_e2e_DNA_wall.dat` | free DNA duplex, 1D metadynamics |
| `plumed_meta_2D_1Nhp6A_large.dat` | apo Nhp6A, 2D metadynamics |
| `plumed_monitor_1Nhp6A.dat`, `plumed_monitor_2Nhp6A.dat`, `plumed_monitor_DNA.dat` | unbiased equilibrium runs: same collective variables, no bias and no walls |

Each file defines the collective variables used throughout: the DNA end-to-end distance
(`de2e`), the DNA bending angle (`theta`), the number of IDR–DNA contacts (`coord_tail_dna`), the
DNA duplex contact map (`cmap`) and, for the apo system, the fraction of native core contacts
(`q_core`) and IDR–core contacts (`n_tail_core`). Multiple-walker runs read and write shared
hills through `WALKERS_DIR`.

### `analysis/scripts/`

| Script | What it does |
|---|---|
| `metad_convergence.py` | Merges walker `HILLS`, generates free-energy surfaces vs simulation time, reports drift, hill-height decay and CV recrossings |
| `fes_convergence.py` | 1D marginal PMF convergence figures and mean-absolute-deviation vs time |
| `fes_marginal_meanstd.py` | 1D marginals with a mean ± standard-deviation band across time snapshots |
| `reweight_metad.py` | Time-dependent c(t) (Tiwary–Parrinello) reweighting from `HILLS` + colvar files |
| `reweight_angle.py` | Final-bias reweighting of unbiased (spectator) observables, e.g. the DNA bending angle |
| `reweight_qbh.py` | 2D PMF along a Best–Hummer fraction of native contacts for the apo system |
| `dna_bending_wall_compare.py` | Bending-angle and end-to-end PMFs compared across free DNA and the complexes |
| `apo_rmsf.py` | Reweighted per-residue Cα RMSF of the folded apo HMG box |
| `equil_monitor.py`, `equil_timeseries.py` | Thermodynamic and structural time series for the unbiased equilibrium runs |
| `select_snapshots.py` | Segments a 2D free-energy surface into basins and extracts a medoid (representative) structure for each |
| `pbc_mindist_check.sh`, `pbc_summary.py` | Minimum-image (periodic-boundary) checks on the solute |
| `figure5_data.py`, `figure5_plot.py` | Compute and write the tabulated data behind each figure panel, and render the panels |
| `make_figure5.py` | Earlier single-file figure assembler, superseded by the two scripts above |

`figure5_data.py` performs the expensive work once and writes plain-text tables; the notebooks
only read those tables, so panels can be restyled without touching the trajectories.

### `analysis/notebooks/`

- `analysis_figure5.ipynb` — main-text figure panels (apo free-energy landscape and helicity,
  representative apo structures, DNA bending in the free duplex vs the complex, equilibrium time
  series for the 2:1 complex).
- `analysis_figure_supp.ipynb` — supplementary panels comparing the wild type with the
  phosphomimetic variants.

### `structures_states/`

PDB files of the representative (medoid) structures of the free-energy basins shown in the
figures: the folded apo state, the partially unstructured apo state, and a bent-DNA state of the
1:1 complex.

## Reproducing the simulations

1. Prepare a system: `setup/setup_amber_<system>.sh` followed by `setup/setup_<system>.sh`
   (requires GROMACS; the bundled force field is used directly).
2. Copy the matching `plumed/` input next to the production `.tpr`.
3. Launch six walkers of `gmx mdrun ... -plumed plumed_meta_*.dat`, each in its own subdirectory
   sharing a common `WALKERS_DIR`. Biased runs were 300 ns per walker for the complexes, 100 ns
   for free DNA and 1000 ns for the apo protein; unbiased runs were 250 ns per replica.
4. Check convergence with `analysis/scripts/metad_convergence.py` and
   `fes_convergence.py`, and the absence of minimum-image artefacts with
   `analysis/scripts/pbc_mindist_check.sh`.
5. Reweight and build the free-energy profiles with the `reweight_*.py` scripts, then
   `figure5_data.py`, and draw the panels from the notebooks.

Requirements: GROMACS 2024.2 patched with PLUMED 2.9 (for the simulations and for
`plumed sum_hills`/`plumed driver`), and Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`,
`seaborn` and `mdtraj` (for the analysis).

## Data not included

- Raw metadynamics output (`HILLS`, colvar time series) and trajectories for all biased runs.
- Raw trajectories of the unbiased equilibrium runs.
- Derived free-energy tables, which are regenerated by the scripts above once the raw output is
  available.

These are available from the corresponding author on request.

