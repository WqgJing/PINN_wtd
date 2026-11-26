# Water-table depth from soil moisture at several depths

Goal: recover the water-table depth zb(t) above a shallow water table from soil-moisture sensors at several depths,
with known soil hydraulic properties, WITHOUT the surface flux and WITHOUT an initial soil profile.

Two inversions of the same problem are included:

- **PINN** (`scripts/train_profile.py`, `src/wpinn/model.py`): a physics-informed neural network represents the
  pressure head between the top sensor and the water table (h = 0 at the water table by construction); the
  water-table path is a daily cubic spline whose coefficients are trained together with the network. All sensors,
  the top one included, are treated as data; no initial profile is needed.
- **Classical FEM inversion** (`scripts/classical_invert.py`, `src/wpinn/fem_top.py`): a finite-element Richards
  solver on the column from the top sensor to the water table, with the top sensor as a fixed-moisture (Dirichlet)
  boundary and a hydrostatic start; the same spline coefficients are fitted by Levenberg-Marquardt to the deeper
  sensors. Optionally the top sensor's offset and drift are fitted as two extra unknowns (`--est_top`).

Default settings make both inversions work from a starting water table well away from the answer (tested from
starts up to 3 m with the water table at 0.7 - 1.1 m):

- PINN (`configs/wt4_d35_55.yaml`): `net.head_form: scale` writes the pressure as the hydrostatic profile scaled by
  exp(N), so it stays negative above the water table and the column cannot saturate (a saturated column has no
  gradient and stalls training); `train.coef_lr_factor: 1.0` and `train.coef_hold_frac: 0.5` let the water table
  travel several metres during training. The original form is `head_form: shift`, `coef_lr_factor: 0.1`,
  `coef_hold_frac: 0`.
- FEM inversion (`scripts/classical_invert.py`): `--max_step 1.0` caps the change of any spline coefficient per
  iteration and `--damp_floor 1e-3` damps coefficients the sensors barely see, so the fit cannot jump to a far
  water table; `--max_step 0 --damp_floor 0 --max_iter 30` is the original fit.

Synthetic truth records are made with the finite-element solver (`scripts/make_synthetic.py`), optionally with a
different root distribution or with soil varying with depth (`src/wpinn/fem_hetero.py`). Sensor errors (offsets,
drift, gain, noise, spikes, gaps, clock shifts, sparse sampling) and a standard quality control are options of the
data builder in `scripts/train_profile.py`, shared by both inversions.

## Layout

| folder | content | in git |
|---|---|---|
| `src/wpinn/` | model, physics, solver wrappers | yes |
| `scripts/` | entry points run through `hpc/`, plotting scripts | yes |
| `configs/` | run and synthetic-record configurations (YAML) | yes |
| `tests/` | unit tests | yes |
| `hpc/` | SLURM scripts and submission helpers | yes |
| `data/` | input data | no |
| `output/`, `runs/`, `logs/` | results, trained networks, job logs | no |

## Notes

- Every Python run goes through `sbatch hpc/run_script_cpu.slurm` or `hpc/run_script_gpu.slurm`.
- The finite-element solver is loaded read-only from a separate `wtd_pipeline` checkout (path in the synthetic
  configs); code copied from an earlier project is marked with its origin in the file header.
- Plots: line plots only, axis labels with units, a legend only for two or more lines.
