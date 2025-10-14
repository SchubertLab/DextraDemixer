import sys
import hashlib
import argparse
import os
import yaml

sys.path.append("../../")

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from dextrademixer.utils import DextramerSimulator, convert_str_to_bool_and_none

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pull obs/var columns from individual modalities.*",
    category=FutureWarning,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", type=str, help="Scenario name", default="scenario_test")

    args = parser.parse_args()
    args = convert_str_to_bool_and_none(args)

    save_path = os.path.join('benchmarks', args.scenario, 'simulation')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join('benchmarks', args.scenario, 'config.yaml'), "r") as f:
        sim_config = yaml.safe_load(f)
    param_grid = ParameterGrid(sim_config)
    print("Number of parameter sets: ", len(param_grid))

    for params in tqdm(param_grid):
        if 'nof_clones' not in params and 'frac_clone' in params:
            params['nof_clones'] = int(params['frac_clone'] * params['total_cell'])
        params = argparse.Namespace(**params)
        sim_config = (f"{params.total_cell}_{params.nof_clones}_{params.p_binding_outlier}_{params.binding_ratio}_"
                      f"{params.use_clonotype_cov}_{params.mean_inc}_{params.var_inc}_{params.i}")
        if os.path.exists(os.path.join(save_path, f"{sim_config}.h5mu")):
            print(f"Simulation with config {sim_config} already exists, skipping...")
            continue

        # Use sim_hash to ensure reproducibility, but have variance in seed
        sim_hash = hashlib.sha256(sim_config.encode('utf-8')).digest()
        seed = int.from_bytes(sim_hash[:4], byteorder='little')

        plt.ioff()
        sim = DextramerSimulator()
        mdata, fig = sim.simulate_pmhc_data_from_distribution(total_cells=params.total_cell,
                                                              nof_clones=params.nof_clones,
                                                              p_binding_outlier=params.p_binding_outlier,
                                                              binding_ratio=params.binding_ratio,
                                                              mean_inc=params.mean_inc,
                                                              var_inc=params.var_inc,
                                                              simulate_neg_control=True,
                                                              use_clonotype_cov=params.use_clonotype_cov,
                                                              plot_data=True,
                                                              rng_key=seed)

        mdata.write(os.path.join(save_path, f"{sim_config}.h5mu"))
        fig.savefig(os.path.join(save_path, f"{sim_config}.pdf"))
        plt.close()

    print(f"Finished simulations for scenario {args.scenario}.")


if __name__ == "__main__":
    main()
