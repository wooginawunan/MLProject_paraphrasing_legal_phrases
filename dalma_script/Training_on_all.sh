#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:50:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=100GB
#SBATCH --job-name=visual
#SBATCH --mail-type=END
#SBATCH --mail-user=nan.wu@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH -o home/nw1045/ML_Project/job.%J.out
#SBATCH -e home/nw1045/ML_Project/job.%J.err
#source activate pythonENVfor_MLProject
# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be in effective! ****
#
# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
module load anaconda/2-4.1.1
source activate pythonENVfor_MLProject

# You may need to load gcc here .. This is application specific
# module load gcc 

# Replace this with your actual command. 'serial-hello-world' for examp
cd home/nw1045/ML_Project/
python model_train.py