#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -N DataAugmentation

cd /rds/general/user/ft824/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 03-Data_Augmentation.py
