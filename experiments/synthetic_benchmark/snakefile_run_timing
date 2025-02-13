from snakemake.utils import min_version
min_version("6.0")


# Configuration
tools = ["dextramixer"] 
reps = 5

# Scenario 4: Timing analyse on reserved node
scenario = {
    "cell_list": [2000, 5000, 1000, 25000, 50000],
    "frac_clone_list": [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
    "p_list": [0.1],
    "cov_list": [None],
    "p_outlier_list": [0], 
    "mean_fold_increase": [5],
    "var_fold_increase": [2], 
}


def aggregate_simulation():
    sim_name = [f"ncell{cell}_nclone{int(clone*cell)}_poutlier{po}_pbinder{p}_negctr1_cov{cov}_meaninc{mean}_varinc{var}_rep{i}"
                for cell in scenario["cell_list"]
                for clone in scenario["frac_clone_list"]
                for po in scenario["p_outlier_list"]
                for p in scenario["p_list"]
                for cov in scenario["cov_list"]
                for mean in scenario["mean_fold_increase"]
                for var in scenario["var_fold_increase"]
                for i in range(reps)]
    return sim_name


rule all:
    input:
        "results/aggregated_results_timing.csv"

rule run_simulation:
    priority: 30
    input:
        # No input files required for this script
    output:
        h5mu = "simulation/sim_ncell{cell}_nclone{clone}_poutlier{po}_pbinder{p}_negctr1_cov{cov}_meaninc{mean}_varinc{var}_rep{i}.h5mu",
        pdf = "simulation/sim_ncell{cell}_nclone{clone}_poutlier{po}_pbinder{p}_negctr1_cov{cov}_meaninc{mean}_varinc{var}_rep{i}.pdf"
    params:
        reps = reps
    conda:
        "environment.yaml"
    shell:
        "python create_data_mean_variance_fold_increase.py {output.h5mu} {output.pdf} "
        "{wildcards.cell} {wildcards.clone} {wildcards.po} {wildcards.p} {wildcards.cov} "
        "{wildcards.mean} {wildcards.var} {params.reps}"

rule run_tool:
    priority: 20
    input:
        "simulation/sim_ncell{cell}_nclone{clone}_poutlier{po}_pbinder{p}_negctr1_cov{cov}_meaninc{mean}_varinc{var}_rep{i}.h5mu"
    output:
        "prediction/{tool}_ncell{cell}_nclone{clone}_poutlier{po}_pbinder{p}_negctr1_cov{cov}_meaninc{mean}_varinc{var}_rep{i}.pred.csv"
    conda:
        "environment.yaml"
    shell:
        "python run_{wildcards.tool}.py {input} {output}"

rule aggregate_results:
    priority: 10
    input:
        expand("prediction/{tool}_{sim_name}.pred.csv",
               tool=tools,
               sim_name=aggregate_simulation())
    output:
        "results/aggregated_results.csv"
    conda:
        "environment.yaml"
    shell:
        "python aggregate_results.py prediction {output}"
