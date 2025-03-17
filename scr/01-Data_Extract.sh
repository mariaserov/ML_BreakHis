#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N DataLoad

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 01-Data_Extract.py
