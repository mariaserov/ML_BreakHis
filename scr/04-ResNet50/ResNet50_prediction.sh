#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=200gb
#PBS -N ResNet50_prediction


cd /rds/general/user/ft824/home/ML_BreakHis/scr

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate breakhis

python resnet50_prediction.py