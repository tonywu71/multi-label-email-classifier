#!/bin/bash
#SBATCH --job-name=ml_groundtruth
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1 
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_med
#SBATCH --mail-user=tony.wu@student-cs.fr
#SBATCH --mail-type=END

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate ml

# To compute in the submission directory
cd $WORKDIR
cd ml_groundtruth/

# Run python script
python ml_matthias_estimators.py