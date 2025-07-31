#!/bin/sh

#SBATCH --job-name=emo-codes
#SBATCH --mem=180g        
#SBATCH --time=4:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e emo-codes-%j.err
#SBATCH -o emo-codes-%j.out

cd /users/9/reine097/projects/auto-label-emotions/ || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/auto-label-emotions/src"

/usr/bin/env /users/9/reine097/projects/auto-label-emotions/.venv/bin/python \
  /users/9/reine097/projects/auto-label-emotions/src/enhanced_pipeline/main.py
