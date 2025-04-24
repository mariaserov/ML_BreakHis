#PBS -l walltime=3:00:00
#PBS -l select=1:ncpus=8:mem=20gb
#PBS -N DenseNet121_holdout_binary_1

cd /rds/general/user/js4124/home/ML_BreakHis/DenseNet

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate ml_py

python DenseNet121_holdout_binary.py 