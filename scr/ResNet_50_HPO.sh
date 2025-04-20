#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=200gb
#PBS -N ResNet50HPO
#PBS -J 1-24

cd /rds/general/user/ft824/home/ML_BreakHis/scr

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate breakhis

python ResNet_50_HPO.py $PBS_ARRAY_INDEX