# Water-table depth from soil moisture at several depths with a physics-informed neural network

Goal: recover the water-table depth zb(t) above a shallow water table from soil-moisture sensors at several depths,
with known soil hydraulic properties, WITHOUT the surface flux and WITHOUT an initial soil profile. A physics-informed
neural network represents the pressure head between the top sensor and the water table; the water-table path is a
spline whose coefficients are trained together with the network.

Why a PINN here: a numerical solver needs the upper boundary (flux or pressure) and the initial profile; a PINN can
replace both by the sensor data inside the domain. No published inversion of the water table from multi-depth soil
moisture was found in a quick search (23 Nov 2026; see `tasks/todo.md`).

Status: set up on 23 Nov 2026. Synthetic study first; nothing has been run. The plan with fixed pass criteria is in
`tasks/todo.md` and is approved before any run.

## Layout

| folder | content | in git |
|---|---|---|
| `src/wpinn/` | model code | yes |
| `scripts/` | entry points run through `hpc/` | yes |
| `configs/` | run configurations (YAML) | yes |
| `tests/` | unit tests | yes |
| `hpc/` | SLURM scripts (`run_script_cpu.slurm`, `run_script_gpu.slurm`) | yes |
| `tasks/` | `todo.md`: plans, decisions, results | yes |
| `data/` | input data (untrusted; one new folder per source) | no |
| `output/`, `runs/`, `logs/` | results, trained networks, job logs | no |

## Rules

- Every Python run goes through `sbatch hpc/run_script_cpu.slurm` or `hpc/run_script_gpu.slurm`; nothing is run on the
  login node. Python environment: the existing `pinn_torch` one (absolute path in the SLURM scripts).
- Code taken from `../pinn/zb_inverse` is copied here with its origin noted in the file header; that project (paused)
  and `../wtd_pipeline` (read only) are not modified from here.
- Plots: line plots only, no titles or annotations; axis labels with units; a legend only for two or more lines.
- Lessons from mistakes go to `/storage/project/r-rbras6-0/jjason3/tasks/lessons.md`.
