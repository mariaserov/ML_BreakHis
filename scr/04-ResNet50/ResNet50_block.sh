#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N ResNet_50_block
#PBS -J 1-6

cd /rds/general/user/ft824/home/ML_BreakHis/scr

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate breakhis

python ResNet50_block.py $PBS_ARRAY_INDEX