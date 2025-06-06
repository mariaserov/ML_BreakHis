#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:mem=300gb
#PBS -N ConvnextV2_FinalModel_Binary

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 04g-ConvNeXT-V2_final_binary.py 