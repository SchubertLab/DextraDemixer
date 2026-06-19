import yaml
import itertools

# Configuration
SCENARIO = config.get("SCENARIO", "test")
RAM_PER_CELL_NUM = float(config.get("RAM_PER_CELL_NUM", 0.05))
SCALING_TEST = str(config.get("SCALING_TEST", False)).lower() in {"1", "true", "yes", "y"}
NODE = config.get("NODE", None)
PREEMPTIBLE = str(config.get("PREEMPTIBLE", False)).lower() in {"1", "true", "yes", "y"}

with open(f"benchmarks/{SCENARIO}/config.yaml") as f:
    config_yaml = yaml.safe_load(f)

print(f"Running scenario: {SCENARIO} on Node={NODE} as scaling test={SCALING_TEST} and RAM per cell = {RAM_PER_CELL_NUM} on {'preemptible' if PREEMPTIBLE else 'non-preemptible'} cluster")
print(config_yaml)

target_files = []
keys = config_yaml.keys()
values = config_yaml.values()
for combo in itertools.product(*values):
    kwargs = dict(zip(keys, combo))
    # Filter out configurations where N * p < 1
    if float(kwargs["total_cells"]) * float(kwargs["binding_ratio"]) >= 1:
        target_files.append(
            f"benchmarks/{SCENARIO}/csv/{kwargs['model_config']}-"
            f"{kwargs['total_cells']},0.4,{kwargs['p_binding_outlier']},{kwargs['binding_ratio']},False,{kwargs['mean_inc']},None,{kwargs['i']}.csv"
        )

print(target_files)

rule all:
    input:
        f"benchmarks/{SCENARIO}/aggregated_results.csv"


rule aggregate_results:
    input:
        lambda wc: target_files
    output:
        "benchmarks/{scenario}/aggregated_results.csv"
    resources:
        c=1,
        mem="8000M",
        node="",
        qos="cpu_preemptible" if PREEMPTIBLE else "cpu_normal",
    shell:
        r"""
        apptainer exec --bind "$PWD/../..":"$PWD/../.." \
                       --pwd "$PWD" dextrademixer.sif \
        python aggregate_results.py \
            --f_ins {input} \
            --f_out {output}
        """

rule run_simulation:
    priority: 30
    output:
        "benchmarks/{scenario}/simulation/{N},0.4,{po},{p},False,{mean_inc},None,{i}.h5mu"
    resources:
        c=1,
        mem=lambda wc: str(float(wc.N) * RAM_PER_CELL_NUM + 1500) + "M",
        node="",  # always use all nodes
        qos="cpu_preemptible" if PREEMPTIBLE else "cpu_normal",
    shell:
        r"""
        apptainer exec --bind "$PWD/../..":"$PWD/../.." \
                       --pwd "$PWD" dextrademixer.sif \
        python simulate_data.py \
            --f_out {output} \
            --total_cells {wildcards.N} \
            --frac_clones 0.4 \
            --p_binding_outlier {wildcards.po} \
            --binding_ratio {wildcards.p} \
            --mean_inc {wildcards.mean_inc} \
            --i {wildcards.i}
        """


rule run_dextrademixer:
    priority: 20
    input:
        "benchmarks/{scenario}/simulation/{N},0.4,{po},{p},False,{mean_inc},None,{i}.h5mu"
    output:
        protected(
            "benchmarks/{scenario}/csv/Dextra{model_config}-"  # wildcard cannot be empty, therefore have to use a small hack here
            "{N},0.4,{po},{p},False,{mean_inc},None,{i}.csv",
        )
    params:
        neg_ctrl_key=lambda wc: "neg_control" if wc.model_config == "Demixer+neg." else "None"
    resources:
        c=1,
        node=NODE if NODE is not None else "",
        qos="cpu_preemptible" if PREEMPTIBLE else "cpu_normal",
        mem=lambda wc: str(float(wc.N) * RAM_PER_CELL_NUM + 3000) + "M",
        job_token=1000 if SCALING_TEST else 1,  # allow only 1 concurrent job for scaling test, by using 1000 out of 1000 tokens. For non-scaling test, allow all jobs to run concurrently. Need to set job tokens to 1000
    shell:
        r"""
        apptainer exec --bind "$PWD/../..":"$PWD/../.." \
                       --pwd "$PWD" dextrademixer.sif \
        python run_dextrademixer.py \
            --f_in {input} \
            --f_out {output} \
            --neg_ctrl_key {params.neg_ctrl_key} \
            --scaling_test {SCALING_TEST}
        """


rule run_beam:
    priority: 20
    input:
        "benchmarks/{scenario}/simulation/{N},0.4,{po},{p},False,{mean_inc},None,{i}.h5mu"
    output:
        protected(
            "benchmarks/{scenario}/csv/BEAM-"  # wildcard cannot be empty, therefore have to use a small hack here
            "{N},0.4,{po},{p},False,{mean_inc},None,{i}.csv",
        )
    resources:
        c=1,
        mem="1000M",
        qos="cpu_preemptible" if PREEMPTIBLE else "cpu_normal",
        node="",
    shell:
        r"""
        apptainer exec --bind "$PWD/../..":"$PWD/../.." \
                       --pwd "$PWD" dextrademixer.sif \
        python run_beam.py \
            --f_in {input} \
            --f_out {output}
        """
