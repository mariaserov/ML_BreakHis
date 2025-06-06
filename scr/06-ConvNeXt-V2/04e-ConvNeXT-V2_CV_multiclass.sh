#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=8:mem=300gb
#PBS -N ConvnextV2_CV_multiclass
#PBS -J 1-5

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 04e-ConvNeXT-V2_CV_multiclass.py $PBS_ARRAY_INDEX