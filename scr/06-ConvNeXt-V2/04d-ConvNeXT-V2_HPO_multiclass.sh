#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -N ConvnextV2HPOmulticlass
#PBS -J 1-24

cd /rds/general/user/ms7024/home/ML_BreakHis/scr

module load anaconda3/personal
source activate breakhis

python 04d-ConvNeXT-V2_HPO_multiclass.py $PBS_ARRAY_INDEX