from snakemake.utils import min_version
min_version("6.0")


# Configuration
SCENARIO = config.get("SCENARIO", "scenario1")
NUM_THREADS = int(config.get("NUM_THREADS", 16))
RAM_PER_THREAD = int(config.get("RAM_PER_THREAD", 16))
USE_MP = config.get("USE_MP", "False") == "True"
if not USE_MP:
    NUM_THREADS = 1

TOOL_CONFIGS = ([f"{model_type}-{mode}-{neg_ctrl_key}-{ir_clone_key}-{alpha_model}-{hyperprior}-{lr}"
                for model_type in ["mixturemodelkmeans"]
                for mode in ["I", "C"]
                for neg_ctrl_key in [None, "neg_control"]
                for ir_clone_key in [None, "clone_id"]
                for alpha_model in ["kmeans", "overdispersion"]
                for hyperprior in [1e2, 1e1, 1e0]
                for lr in [1e-2]
                if ((mode == "C" and ir_clone_key is not None) or mode != "C")
                   and
                   ((alpha_model == "kmeans" and hyperprior > 1e0) or (alpha_model == "overdispersion" and hyperprior <= 1e1))
                ])

# TOOL_CONFIGS = ([f"{model_type}_{mode}_{neg_ctrl_key}_{ir_clone_key}_{alpha_model}_{hyperprior}_{lr}"
#                 for model_type in ["mixturemodelkmeans"]
#                 for mode in ["I"]
#                 for neg_ctrl_key in ["neg_control"]
#                 for ir_clone_key in ["clone_id"]
#                 for alpha_model in ["overdispersion"]
#                 for hyperprior in [1e-1, 1e-2]
#                 for lr in [1e-2]
#                 if (mode == "C" and ir_clone_key is not None) or mode != "C"
#                 ])

rule all:
    input:
        expand("benchmarks/{scenario}/aggregated_results.csv",scenario=SCENARIOS)


rule run_simulation:
    priority: 30
    input:
        # No input files required for this script
    output:
        "benchmarks/{scenario}/simulation.finished"
    conda:
        "environment.yaml"
    shell:
        "python simulate_scenarios.py --scenario {wildcards.scenario} "
        "&& touch {output}"


rule run_dextrademixer:
    priority: 20
    input:
        "benchmarks/{scenario}/simulation.finished"
    output:
        "benchmarks/{scenario}/{model_type}-{mode}-{neg_ctrl_key}-{ir_clone_key}-{alpha_model}-{hyperprior}-{lr}.finished"
    conda:
        "environment.yaml"
    threads:
        min(32, NUM_THREADS)  # Cluster can only request 32 cores at once
    params:
        use_mp=USE_MP
    resources:
        c=min(32, NUM_THREADS),
        mem_mb=RAM_PER_THREAD * 1000 * min(32, NUM_THREADS),
        mem=f"{RAM_PER_THREAD * min(32, NUM_THREADS)}G",
    shell:
        "python run_dextrademixer_mp.py --scenario {wildcards.scenario} --model_type {wildcards.model_type} "
        "--mode {wildcards.mode} --neg_ctrl_key {wildcards.neg_ctrl_key} --ir_clone_key {wildcards.ir_clone_key} "
        "--alpha_model {wildcards.alpha_model} --hyperprior {wildcards.hyperprior} --lr {wildcards.lr} --maxiter 5000 --use_mp {params.use_mp} "
        "&& touch {output}"


rule aggregate_results:
    priority: 10
    input:
        lambda wildcards: expand(
            f"benchmarks/{wildcards.scenario}/{{tool_config}}.finished",
            tool_config=TOOL_CONFIGS
        )
    output:
        "benchmarks/{scenario}/aggregated_results.csv"
    conda:
        "environment.yaml"
    shell:
        "python aggregate_results.py --scenario {wildcards.scenario}"
