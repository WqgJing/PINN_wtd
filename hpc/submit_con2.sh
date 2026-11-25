#!/bin/bash
# Usage: hpc/submit_con2.sh <version tag> <sensor list, e.g. [0]>   Test #2: 8 realisations (noise seed r, contamination
# seed 100 + r), QC on; PINN (H200 + V100 raced), FEM-std and FEM-est per realisation. Ids in logs/race/c2_<tag>_ids.txt.
cd /storage/project/r-rbras6-0/jjason3/wtd_profile_pinn
tag=$1; sens=$2; R=runs/place_4_5-20-40-60; ALL=""
for r in 0 1 2 3 4 5 6 7; do
  S="--set sensors.qc=true --set sensors.noise_seed=$r --set sensors.contamination.seed=$((100 + r)) --set sensors.contamination.sensors=$sens"
  P=runs/c2_${tag}_r${r}_4
  h=$(timeout 120 sbatch --parsable -J wpp_c2${tag}$r -t 06:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
  v=$(timeout 120 sbatch --parsable -J wpp_c2${tag}$r --partition=gpu-v100 --gres=gpu:v100:1 -t 08:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
  f=$(timeout 120 sbatch --parsable -J wpp_fc2${tag}$r -c8 --mem=16G -t 04:00:00 --export=ALL,WPP_THREADS=8 hpc/run_script_cpu.slurm scripts/classical_invert.py --run $R $S --top smooth --rel_tol 1e-3 --pinn $P --out runs/fem_c2_${tag}_r${r}_4)
  e=$(timeout 120 sbatch --parsable -J wpp_ec2${tag}$r -c8 --mem=16G -t 06:00:00 --export=ALL,WPP_THREADS=8 hpc/run_script_cpu.slurm scripts/classical_invert.py --run $R $S --top smooth --rel_tol 1e-3 --est_top --pinn $P --out runs/fest_c2_${tag}_r${r}_4)
  echo "$tag r$r: PINN $h|$v FEM-std $f FEM-est $e"
  ALL="$ALL,$h,$v,$f,$e"
  nohup bash hpc/race.sh "$h,$v" > logs/race/c2_${tag}_r${r}.txt 2>&1 &
done
echo "${ALL#,}" > logs/race/c2_${tag}_ids.txt
