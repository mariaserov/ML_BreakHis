#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=100gb
#PBS -N ResNet_50_multiclass
#PBS -J 1-6

cd /rds/general/user/ft824/home/ML_BreakHis/scr

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate breakhis

python ResNet50_multiclass.py $PBS_ARRAY_INDEX