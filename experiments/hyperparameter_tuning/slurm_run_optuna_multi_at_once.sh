#!/bin/bash


PATH_BASE="$(dirname "$(realpath "$0")")/../.."
PATH_LOGS="."

mkdir -p "${PATH_LOGS}/logs/slurm_logs"
mkdir -p "${PATH_LOGS}/logs/jobs"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="SKM_dex_optuna_${DATE_TIME}"

echo "#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_LOGS}/logs/slurm_logs/${JOB_NAME}.out
#SBATCH -e ${PATH_LOGS}/logs/slurm_logs/${JOB_NAME}.err
#SBATCH -p cpu_p
#SBATCH -t 3-00:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

source ~/.bash_profile
conda activate dextrademixer
cd ${PATH_BASE}/experiments/hyperparameter_tuning

snakemake -s snakefile_run_optuna_multi_at_once --unlock
snakemake -s snakefile_run_optuna_multi_at_once -c4 --profile ${PATH_BASE}/experiments/.slurm/ --cluster-status ${PATH_BASE}/experiments/.slurm/status.py --conda-frontend conda


" > ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
sbatch ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
