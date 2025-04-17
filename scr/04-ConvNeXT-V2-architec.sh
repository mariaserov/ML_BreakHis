#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N ConvNeXTV2
#PBS -J 1-5

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 04-ConvNeXT-V2-architec.py $PBS_ARRAY_INDEX