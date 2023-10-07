#!/bin/bash
#$ -l h_rt=00:05:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.11/3.11.2 cuda/11.7/11.7.1 cudnn/8.8/8.8.1 nccl/2.14/2.14.3-1
export LD_LIBRARY_PATH=/apps/gcc/12.2.0/lib64:/apps/python/3.11.2/lib
source ~/cuda/bin/activate

export PT_CACHE_DIR=$SGE_LOCALDIR

python3 train_model.py --config ../config/training_setup.yaml --urls ../datasets/urls.txt