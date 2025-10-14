#!/bin/bash
# Usage:
#   bash slurm_run_smk.sh                # cpu_normal
#   bash slurm_run_smk.sh --preemptible  # cpu_preemptible


PATH_BASE="$(dirname "$(realpath "$0")")/../.."
PATH_LOGS="."

PROFILE_DIR="${PATH_BASE}/experiments/.slurm"
[[ "${1-}" == "--preemptible" ]] && PROFILE_DIR="${PATH_BASE}/experiments/.slurm_preemptible"


mkdir -p "${PATH_LOGS}/logs/slurm_logs"
mkdir -p "${PATH_LOGS}/logs/jobs"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="SKM_dex_benchmark_${DATE_TIME}"

echo "#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_LOGS}/logs/slurm_logs/${JOB_NAME}.out
#SBATCH -e ${PATH_LOGS}/logs/slurm_logs/${JOB_NAME}.err
#SBATCH -p cpu_p
#SBATCH -t 1-00:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

source ~/.bash_profile
conda activate dextrademixer
cd ${PATH_BASE}/experiments/synthetic_benchmark

snakemake -s snakefile_run_simulation.smk --unlock
snakemake -s snakefile_run_simulation.smk -c4 --profile ${PROFILE_DIR} --cluster-status ${PROFILE_DIR}/status.py --conda-frontend conda -p


" > ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
sbatch ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
