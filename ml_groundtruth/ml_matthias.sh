#!/bin/bash
#SBATCH --job-name=ml_groundtruth
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_med

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate ml

# To compute in the submission directory
cd $WORKDIR
cd ml_groundtruth/

# Run python script
python ml_matthias.py