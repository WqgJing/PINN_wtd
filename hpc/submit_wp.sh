#!/bin/bash
# Usage: hpc/submit_wp.sh <tag> <record dir under output/>   PINN (H200 + V100 raced) and FEM for noise seeds 0-2 on the
# 4-sensor setup of runs/place_4_5-20-40-60, data from that record. Prints the ids; races logged in logs/race/wp_<tag>_s<seed>.txt
cd /storage/project/r-rbras6-0/jjason3/wtd_profile_pinn
tag=$1; rec=$2; R=runs/place_4_5-20-40-60; ALL=""
for s in 0 1 2; do
  S="--set data.npz=/storage/project/r-rbras6-0/jjason3/wtd_profile_pinn/output/$rec/synthetic.npz --set sensors.noise_seed=$s"
  P=runs/wp_${tag}_s${s}_4
  h=$(timeout 120 sbatch --parsable -J wpp_wp_${tag}$s -t 06:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
  v=$(timeout 120 sbatch --parsable -J wpp_wp_${tag}$s --partition=gpu-v100 --gres=gpu:v100:1 -t 08:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
  f=$(timeout 120 sbatch --parsable -J wpp_fwp_${tag}$s -c8 --mem=16G -t 03:00:00 --export=ALL,WPP_THREADS=8 hpc/run_script_cpu.slurm scripts/classical_invert.py --run $R $S --top smooth --rel_tol 1e-3 --pinn $P --out runs/fem_wp_${tag}_s${s}_4)
  echo "$tag seed $s: PINN H200 $h V100 $v; FEM $f"
  ALL="$ALL,$h,$v,$f"
  nohup bash hpc/race.sh "$h,$v" > logs/race/wp_${tag}_s${s}.txt 2>&1 &
done
echo "${ALL#,}" > logs/race/wp_${tag}_ids.txt
