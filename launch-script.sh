#!/usr/bin/env bash
#PBS -q qgpu
#PBS -N d4cbatch2
#PBS -l select=1:ngpus=1,walltime=20:00:00
#PBS -A OPEN-20-37

cd $HOME/darts4coatnet

#ml cuDNN/8.4.1.50-CUDA-11.7.0 Anaconda3
ml cuDNN/8.2.2.26-CUDA-11.4.1  Anaconda3
#ml cuDNN/8.0.4.30-CUDA-11.1.1 Anaconda3
#ml CUDA/11.1.1-GCC-10.2.0 Anaconda3
#ml cuDNN/8.2.1.32-CUDA-11.3.1 Anaconda3

#source activate darts
source activate latest
#source activate darts-old
#source activate old

#python3 train.py --genotype_file logs/search_arch/20230210-232719/train/genotype_best --batch_size 64 --epochs 400 --init_channels 36 --layers 20 --cutout --auxiliary

# 20230207-002742 test this arch

python3 search_cell.py --seed 0 --batch_size 64