#!/bin/sh

#SBATCH --job-name=facial-expression-training # job name

#SBATCH --mem=180g        
#SBATCH --time=1:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e facial-expression-training-%j.err
#SBATCH -o facial-expression-training-%j.out

cd /users/9/reine097/projects/auto-label-emotions || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/auto-label-emotions/src"
pwd
/users/9/reine097/projects/auto-label-emotions/.venv/bin/python \
  /users/9/reine097/projects/auto-label-emotions/src/cdni/pytorch_image_classifier_png_data.py 
