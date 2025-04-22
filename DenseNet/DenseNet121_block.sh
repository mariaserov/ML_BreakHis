#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -N DenseNet121_block
#PBS -J 1-8

cd /rds/general/user/js4124/home/ML_BreakHis/DenseNet

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate ml_py

python DenseNet121_block.py $PBS_ARRAY_INDEX
