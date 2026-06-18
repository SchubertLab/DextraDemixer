#!/bin/bash
# Usage examples:
#   bash slurm_benchmark.sh --scenario test
#   bash slurm_benchmark.sh --scenario synth_benchmark
#   bash slurm_benchmark.sh --scenario scaling --node cpusrv100 --scaling_test --no-preemptible
#   bash slurm_benchmark.sh --scenario dropout

set -euo pipefail

# ---- Default values
SCENARIO="test"
RAM_PER_CELL_NUM=0.05
ADDITIONAL_FLAGS=""
NODE=""
SCALING_TEST=False
PREEMPTIBLE=True

# ---- Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --ram)
            RAM_PER_CELL_NUM="$2"
            shift 2
            ;;
        --no-preemptible)
            PREEMPTIBLE=False
            shift
            ;;
        --scaling_test)
            SCALING_TEST=True
            # PROFILE_SUFFIX=".slurm_one_node"
            shift
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --additional_flags)
            ADDITIONAL_FLAGS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash $0 [--scenario NAME] [--ram MB_per_cell] [--no-preemptible] [--additional_flags 'QUOTED_FLAGS'] [--node NODE_NAME] [--scaling_test]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

PATH_BASE="$(dirname "$(realpath "$0")")/../.."
PATH_LOGS="."
PROFILE_DIR="${PATH_BASE}/experiments/.slurm"


mkdir -p "${PATH_LOGS}/logs/slurm_logs"
mkdir -p "${PATH_LOGS}/logs/jobs"

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
JOB_NAME="SKM_${SCENARIO}_${DATE_TIME}"

echo "Submitting job ${JOB_NAME} to Slurm with scenario=${SCENARIO}, RAM per cell=${RAM_PER_CELL_NUM}MB, node=${NODE}, scaling_test=${SCALING_TEST}, additional_flags='${ADDITIONAL_FLAGS}'"

cat > "${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd" <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${PATH_LOGS}/logs/${JOB_NAME}.out
#SBATCH -e ${PATH_LOGS}/logs/${JOB_NAME}.err
#SBATCH -p cpu_p
#SBATCH -t 1-00:00:00
#SBATCH -c 4
#SBATCH --mem=10G
#SBATCH --qos=cpu_normal
#SBATCH --nice=0

source ~/.bash_profile
conda activate dextrademixer
cd ${PATH_BASE}/experiments/synthetic_benchmark

snakemake -s snakefile_benchmark.smk --unlock --config SCENARIO=${SCENARIO} RAM_PER_CELL_NUM=${RAM_PER_CELL_NUM} NODE=${NODE} SCALING_TEST=${SCALING_TEST} PREEMPTIBLE=${PREEMPTIBLE}

snakemake -s snakefile_benchmark.smk \\
    -c4 \\
    --profile ${PROFILE_DIR} \\
    --cluster-status ${PROFILE_DIR}/status.py \\
    --conda-frontend conda \\
    --config SCENARIO=${SCENARIO} RAM_PER_CELL_NUM=${RAM_PER_CELL_NUM} NODE=${NODE} SCALING_TEST=${SCALING_TEST} PREEMPTIBLE=${PREEMPTIBLE} \\
    --jobs 1000 \\
    --resources job_token=1000 \\
    -p \\
    ${ADDITIONAL_FLAGS}
EOF

sbatch ${PATH_LOGS}/logs/jobs/${JOB_NAME}.cmd
