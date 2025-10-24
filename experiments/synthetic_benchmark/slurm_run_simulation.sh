#!/bin/bash
# Usage examples:
#   bash slurm_run_smk.sh                # cpu_normal
#   bash slurm_run_smk.sh --preemptible  # cpu_preemptible
#   bash slurm_run_smk.sh --scenario scenarioX --threads 8 --ram 32 --no-mp

set -euo pipefail

# ---- Default values
SCENARIO="scenario1"
NUM_THREADS=16
RAM_PER_THREAD=16
USE_MP=True
PROFILE_SUFFIX=".slurm"

# ---- Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --ram)
            RAM_PER_THREAD="$2"
            shift 2
            ;;
        --preemptible)
            PROFILE_SUFFIX=".slurm_preemptible"
            shift
            ;;
        --no-mp)
            USE_MP=False
            shift
            ;;
        -h|--help)
            echo "Usage: bash $0 [--scenario NAME] [--threads N] [--ram GB] [--preemptible] [--no-mp]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---- Paths
PATH_BASE="$(dirname "$(realpath "$0")")/../.."
PATH_LOGS="."
PROFILE_DIR="${PATH_BASE}/experiments/${PROFILE_SUFFIX}"


mkdir -p "${PATH_LOGS}/logs/slurm_logs"
mkdir -p "${PATH_LOGS}/logs/jobs"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="SKM_dex_benchmark_${SCENARIO}_${DATE_TIME}"

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
snakemake -s snakefile_run_simulation.smk -c4 --profile ${PROFILE_DIR} --cluster-status ${PROFILE_DIR}/status.py --conda-frontend conda -p --config SCENARIO=${SCENARIO} NUM_THREADS=${NUM_THREADS} RAM_PER_THREAD=${RAM_PER_THREAD} USE_MP=${USE_MP}



" > ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
sbatch ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
