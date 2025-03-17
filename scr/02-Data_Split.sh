#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=1:mem=10gb
#PBS -N DataSplit

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 02-Data_Split.py
