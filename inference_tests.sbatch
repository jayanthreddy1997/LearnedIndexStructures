#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:60:00
#SBATCH --mem=16GB
#SBATCH --job-name=inference_tests
#SBATCH --output=output/tests.out
#SBATCH --gres=gpu:v100:1

singularity exec \
            --nv --overlay /scratch/nr2686/idls_env/overlay-15GB-500K.ext3 \
            /scratch/work/public/singularity/ubuntu-20.04.1.sif \
            /bin/bash -c "cd /home/nr2686/workspace/idls/LearnedIndexStructures;
            source /ext3/env.sh;
            python inference_time_tests.py"
