#!/bin/bash


PATH_BASE="$(dirname "$(realpath "$0")")/../.."
PATH_LOGS="${PATH_BASE}/logs/slurm_logs"

mkdir -p "${PATH_LOGS}"
mkdir -p "${PATH_BASE}/logs/jobs"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="SKM_simSyn_${DATE_TIME}"

echo "#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_BASE}/logs/slurm_logs/${JOB_NAME}.out
#SBATCH -e ${PATH_BASE}/logs/slurm_logs/${JOB_NAME}.err
#SBATCH -p cpu_p
#SBATCH -t 10-00:00:00
#SBATCH -c 8 
#SBATCH --mem=24G
#SBATCH --qos=cpu_long
#SBATCH --nice=10000

source ~/.bash_profile
conda activate dextraDemixer 
cd ${PATH_BASE}/experiments/synthetic_benchmark

snakemake aggregate_results -s snakefile_run_simulation -c8 --profile ${PATH_BASE}/experiments/.slurm/ --cluster-status ${PATH_BASE}/experiments/.slurm/status.py --conda-frontend conda


" > ${PATH_BASE}/logs/jobs/${JOB_NAME}.cmd
sbatch ${PATH_BASE}/logs/jobs/${JOB_NAME}.cmd
