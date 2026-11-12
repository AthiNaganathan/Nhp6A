# Nhp6A–DNA metadynamics — simulation and analysis package

Input structures, force-field files, simulation inputs and the analysis/plotting code behind the
molecular dynamics results reported in the accompanying paper on the yeast HMG-box protein Nhp6A
and its complexes with DNA.

This is a **methods and code package**: everything needed to set up, launch and analyse the
simulations is here. It also contains the processed free-energy and time-series tables behind
every simulation figure panel, so all panels can be redrawn without rerunning anything. The raw
trajectories, metadynamics `HILLS` and collective-variable time series are too large to
distribute and are not included (see [Data not included](#data-not-included)).

The package was updated for the revised manuscript. It adds the IDR-only and Δ2-10 complexes, the
pulled-start (m-state / mM) unbiased simulations, the figure tables, and a single notebook that
draws every simulation panel.

## Systems

All complexes use a 15-bp B-DNA duplex and, unless stated otherwise, the 93-residue Nhp6A monomer
(HMG box plus its disordered N-terminal region, the IDR).

| Name | Composition | Sampling |
|---|---|---|
| `DNA` | free 15-bp duplex | 1D well-tempered metadynamics on the DNA end-to-end distance, 6 walkers |
| `1Nhp6A` | 1:1 Nhp6A–DNA, wild type | 2D well-tempered metadynamics on (IDR–DNA contacts, end-to-end distance), 6 walkers |
| `1S26D`, `1T63D`, `1TM` | 1:1 complexes of the phosphomimetic variants S26D, T63D and the triple mutant S26D/S41D/T63D | as `1Nhp6A` |
| `1Nhp6A_apo_large` | Nhp6A alone (no DNA), enlarged box | 2D well-tempered metadynamics on (fraction of native core contacts, IDR–core contacts), 6 walkers |
| `1IDR` | IDR peptide alone (Nhp6A residues 1–23) bound to DNA | as `1Nhp6A`, 400 ns per walker |
| `1D2_10` | 1:1 complex of the Δ2-10 deletion variant (84 residues; renumbered 1–84, new = WT − 9 for WT 11–93, Met1 kept) | as `1Nhp6A`, 350 ns per walker |
| `2Nhp6A` | 2:1 Nhp6A–DNA, wild type | unbiased equilibrium MD, 3 replicas |
| `1Nhp6A` m-state | 1:1 complex with the HMG-box core pulled off the DNA, IDR left bound | 4 pulled starts × 2 unbiased replicas × 250 ns |
| `2Nhp6A` mM | 2:1 complex with the core of copy B pulled off, copy A untouched | 2 pulled starts × 2 unbiased replicas × 250 ns |

Unbiased equilibrium MD (3 replicas per system) was additionally run for the free DNA duplex and
for the 1:1 and 2:1 wild-type complexes, with PLUMED used in monitor-only mode to record the same
collective variables without applying any bias.

## Simulation protocol

| | |
|---|---|
| Engine | GROMACS 2024.2, patched with PLUMED 2.9.2 |
| Force field | AMBER ff14SB (protein) + parmbsc1 (DNA), supplied as `structures_topology/amber14sb_parmbsc1.ff` |
| Water / ions | TIP3P, 0.15 M KCl, truncated-octahedron box |
| Temperature | 298 K |
| Enhanced sampling | Well-tempered metadynamics, multiple walkers (6), `PACE=500` |
| Bias (complexes and free DNA) | height 2.0 kJ/mol, bias factor 5, σ = 3.0 (contacts) / 0.25 nm (end-to-end distance) |
| Bias (apo protein) | height 1.2 kJ/mol, bias factor 10, σ = 0.025 (contact fraction) / 1.5 (contacts) |
| Equilibration | EM → 1 ns NVT → 1 ns NPT (Berendsen), heavy-atom position restraints 1000 kJ mol⁻¹ nm⁻²; box buffer 1.0 nm for the complexes, 1.5 nm for free DNA and the 2:1 mM probe, 1.2 / 1.7 nm for the apo boxes |
| Unbiased production | `setup/mdp/md_equil.mdp` (C-rescale barostat, velocities generated with a recorded seed) |
| Restraints on the biased runs | soft `LOWER_WALLS` on the DNA inter-strand contact map (duplex integrity, inactive above Q = 0.9) for the DNA-containing systems; `UPPER_WALLS` on core RMSD (0.5 nm) for the apo protein |

Starting structures were built from AlphaFold 3 predictions of the complexes; the raw predictions
and the cleaned-up versions used for system preparation are in `structures_topology/AF_Nhp6A/`.

## Contents

```
structures_topology/    Starting structures and GROMACS topologies, one directory per system,
                        plus AF_Nhp6A/ (AlphaFold models), amber14sb_parmbsc1.ff/ (force field)
                        and mstate_starts/ (full-system start frames of the m-state / mM pulls)
setup/                  System-preparation scripts (pdb2gmx → solvate → genion → EM → NVT/NPT),
                        mdp/ run-parameter files, PLUMED input generation, DNA contact-map
                        definition, PLUMED atom-group extraction
plumed/                 PLUMED input files as run: biased metadynamics for each system and the
                        monitor-only (unbiased) inputs used for the equilibrium simulations
analysis/scripts/       Convergence assessment, reweighting, free-energy/PMF construction,
                        structure selection, m-state analysis and figure generation
analysis/article/       Processed tables behind every figure panel (+ meta.json provenance)
analysis/*.dat          Per-system 1D PMFs (mean ± std) and 2D PMFs used by the panels
analysis/snapshots/     Basin-medoid CV coordinates drawn on the 2D PMF panels
analysis/notebooks/     article_figures.ipynb, which draws every simulation panel from the tables
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
wild-type model. `1IDR/` and `1D2_10/` also contain the cleaned AlphaFold model each system was
built from (`*_DNA_AF_fix.pdb`).

`mstate_starts/` holds the gzipped full-system frames that start the steered pulls:
`1Nhp6A_mstate_start_c{0,1,3,4}.gro.gz` (frames from the unbiased 1:1 runs) and
`2Nhp6A_mstate_start_c{1,3}.gro.gz` (from the unbiased 2:1 runs). `c<k>` is the frame cluster
chosen by `analysis/scripts/mstate_select_frames.py`. The setup scripts expect them unzipped as
`data/mstate/start/c<k>.gro` and `data/mstate_2Nhp6A/start/2Nhp6A_mstate_start_c<k>.gro`.

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
- `setup_amber_eq.sh` — the EM → NVT → NPT chain (protocol of record) for a prepared system,
  including `1IDR` and `1D2_10`.
- `setup_mstate_pull.sh`, `slurm_mstate_pull.sh` — m-state / mM construction: a 40 ns steered pull
  (`MOVINGRESTRAINT`, κ = 1000 kJ mol⁻¹ nm⁻²) that increases the core–DNA centre-of-mass distance
  `d_core` by 2.0 nm, with a soft `LOWER_WALLS` that keeps the IDR on the DNA.
- `setup_mstate_eq.sh` — re-solvates each pulled end frame, equilibrates it with the same protocol
  and launches the unbiased replicas (no bias and no walls).

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
| `plumed_meta_2D_1IDR_wall.dat`, `plumed_meta_2D_1D2_10_wall.dat` | IDR-only and Δ2-10 complexes, 2D metadynamics (same bias settings as the WT) |
| `plumed_monitor_1Nhp6A.dat`, `plumed_monitor_2Nhp6A.dat`, `plumed_monitor_DNA.dat`, `plumed_monitor_1IDR.dat`, `plumed_monitor_1D2_10.dat` | unbiased equilibrium runs: same collective variables, no bias and no walls |
| `plumed_pull_mstate.dat`, `plumed_pull_mstate_2Nhp6A.dat` | steered core-detachment pulls (templates: `setup_mstate_pull.sh` fills in `__D0__`/`__D1__`) |
| `plumed_monitor_mstate.dat`, `plumed_monitor_mstate_2Nhp6A.dat` | force-free monitors for the unbiased m-state / mM replicas |

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
| `mstate_select_frames.py` | Clusters the unbiased holo frames and writes the m-state / mM start frames |
| `mstate_contact_pca.py` | Core–DNA contact PCA, native-contact fractions, bound-mode (native / shifted / non-native) assignment and rebinding analysis of the m-state / mM runs |
| `mstate_pbc_check.py` | Minimum-image check for the detached-core runs (the protein is placed in the same cell as its DNA first) |
| `article_data.py`, `article_plot.py` | Compute and write the tabulated data behind each figure panel (`analysis/article/`), and the shared loading/truncation/saving helpers |
| `figpaths.py` | Output-path routing for figures |

`article_data.py` performs the expensive work once and writes plain-text tables; the notebook
only reads those tables, so panels can be restyled without touching the trajectories.

### `analysis/notebooks/`

`article_figures.ipynb` draws every simulation panel of the paper and the SI, organised by
system: apo Nhp6A; the 1:1 complex with the IDR-only and Δ2-10 controls; the 2:1 complex; the
m-state / mM probes (SI); and the phosphomimetic variants (SI). It replaces the two notebooks of
the first deposit (main text and SI). Each cell is self-contained, finds the
package root (the first ancestor holding `analysis/scripts/`) and writes PDF + PNG to
`figures/article/`. Running the whole notebook from a clean copy of this folder reproduces all
panels from the bundled tables (verified at deposition).

### `structures_states/`

PDB files of the representative (medoid) structures of the free-energy basins shown in the
figures, named by system: the folded apo state and the partially unstructured (H2-shoulder) apo
state (`1Nhp6A_apo_large_helicity_state_{1,2}_medoid_*`), the bent-DNA state of the 1:1 complex
(`1Nhp6A_wall_bend_state_1_medoid`), and the basin medoids of the IDR-only complex (`1IDR_wall_bend_state_{1,2}`) and of
the Δ2-10 complex (`1D2_10_wall_bend_state_{1,2,3}`) on F(n_IDR–DNA, bend). Their coordinates
are in `analysis/snapshots/*_medoids_cvs.dat`.

## Reproducing the simulations

1. Prepare a system: `setup/setup_amber_<system>.sh` followed by `setup/setup_<system>.sh`
   (requires GROMACS; the bundled force field is used directly).
2. Copy the matching `plumed/` input next to the production `.tpr`.
3. Launch six walkers of `gmx mdrun ... -plumed plumed_meta_*.dat`, each in its own subdirectory
   sharing a common `WALKERS_DIR`. Biased runs per walker: free DNA 100 ns; WT complex 300 ns
   (main-text panels) / 350 ns (SI mutant comparison); S26D and T63D 350 ns; TM 300 ns; IDR-only
   400 ns; Δ2-10 350 ns; apo protein 1250 ns. The exact lengths behind each panel are recorded in
   `analysis/article/meta.json`. Unbiased runs were 250 ns per replica.
   For the m-state / mM probes: `setup/setup_mstate_pull.sh`, then `setup/setup_mstate_eq.sh`.
4. Check convergence with `analysis/scripts/metad_convergence.py` and
   `fes_convergence.py`, and the absence of minimum-image artefacts with
   `analysis/scripts/pbc_mindist_check.sh`.
5. Reweight and build the free-energy profiles with the `reweight_*.py` scripts, then
   `article_data.py`, and draw the panels with `analysis/notebooks/article_figures.ipynb`.

Requirements: GROMACS 2024.2 patched with PLUMED 2.9 (for the simulations and for
`plumed sum_hills`/`plumed driver`), and Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib`,
`seaborn`, `mdtraj` and `scikit-learn` (for the analysis).

## Data not included

- Raw metadynamics output (`HILLS`, colvar time series) and trajectories for all biased runs.
- Raw trajectories of the unbiased equilibrium runs.
- Intermediate analysis products (per-time-snapshot FES grids, reweighting weights, contact
  matrices). The final tables the figures are drawn from *are* included (`analysis/article/`,
  `analysis/*.dat`).

These are available from the corresponding author on request.

