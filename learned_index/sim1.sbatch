#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:60:00
#SBATCH --mem=16GB
#SBATCH --job-name=sim1
#SBATCH --output=output/sim1.out
#SBATCH --gres=gpu:1

singularity exec \
            --nv --overlay /scratch/nr2686/idls_env/overlay-15GB-500K.ext3 \
            /scratch/work/public/singularity/ubuntu-20.04.1.sif \
            /bin/bash -c "cd /home/nr2686/workspace/idls/LearnedIndexStructures/learned_index;
            source /ext3/env.sh;
            python main.py"
