#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=20:mem=50gb
#PBS -N DenseNet121_hpo_multi
#PBS -J 0-35

cd /rds/general/user/js4124/home/ML_BreakHis/DenseNet

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate ml_py

python DenseNet121_hpo_multi.py $PBS_ARRAY_INDEX