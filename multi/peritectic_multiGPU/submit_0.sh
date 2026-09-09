#!/bin/bash
#SBATCH --job-name=60K!|0
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --mem=40Gb
#SBATCH --nodes=1
#SBATCH --output=out_0.txt
#SBATCH --error=error.txt
#SBATCH --time=24:0:0
cd /work/karmalab/mingwang/peritectic/TiAg_multiwell/lambda30_alpha2/3D512_2nuclei_T60K
nvcc TiAg.cu -O3 -lm -arch sm_30  -o tiag
stdbuf -oL ./tiag 12000000 200000 0 > out_0.txt
