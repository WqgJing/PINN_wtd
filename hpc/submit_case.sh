#!/bin/bash
# Usage: hpc/submit_case.sh <tag> <seed> <settings...>   One case on the 4-sensor setup: PINN (H200 + V100 raced) and FEM-std.
# Appends "h,v,f" to logs/race/<group>_ids.txt where group = tag without the trailing _s<seed>; race log logs/race/<tag>.txt
cd /storage/project/r-rbras6-0/jjason3/wtd_profile_pinn
tag=$1; seed=$2; shift 2; R=runs/place_4_5-20-40-60; S="$* --set sensors.noise_seed=$seed"; P=runs/${tag}_4
h=$(timeout 120 sbatch --parsable -J wpp_$tag -t 06:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
v=$(timeout 120 sbatch --parsable -J wpp_$tag --partition=gpu-v100 --gres=gpu:v100:1 -t 08:00:00 hpc/run_script_gpu.slurm scripts/train_profile.py --config $R/config.yaml $S --out $P)
f=$(timeout 120 sbatch --parsable -J wpp_f$tag -c8 --mem=16G -t 04:00:00 --export=ALL,WPP_THREADS=8 hpc/run_script_cpu.slurm scripts/classical_invert.py --run $R $S --top smooth --rel_tol 1e-3 --pinn $P --out runs/fem_${tag}_4)
echo "$tag: PINN $h|$v FEM $f"
echo -n "$h,$v,$f," >> logs/race/${tag%_s*}_ids.txt
nohup bash hpc/race.sh "$h,$v" > logs/race/$tag.txt 2>&1 &
