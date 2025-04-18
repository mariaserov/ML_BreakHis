#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=100gb
#PBS -N ResNet50

cd /rds/general/user/ft824/home/ML_BreakHis/scr

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate breakhis

python ResNet50.py