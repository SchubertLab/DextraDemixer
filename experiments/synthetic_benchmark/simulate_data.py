import sys
import hashlib
import argparse
import os

sys.path.append("../../")

from dextrademixer.utils import DextramerSimulator, convert_str_to_bool_and_none

import warnings
warnings.filterwarnings("ignore", message=".*pull obs/var columns from individual modalities.*", category=FutureWarning,)


def main(params):
    if params.f_out is not None:
        base_path = os.path.dirname(params.f_out)
        sim_config = os.path.basename(params.f_out.replace('.h5mu', ''))
        os.makedirs(base_path, exist_ok=True)
    else:
        base_path = 'simulation'
        os.makedirs(base_path, exist_ok=True)
        sim_config = (f"{params.total_cells},{params.nof_clones if params.nof_clones is not None else params.frac_clones},"
                      f"{params.p_binding_outlier},{params.binding_ratio},False,{params.mean_inc},None,{params.i}")
        params.f_out = os.path.join(base_path, f"{sim_config}.h5mu")
    if params.nof_clones is None:
        params.nof_clones = int(params.frac_clones * params.total_cells)

    if os.path.exists(params.f_out):
        print(f"Simulation with config {sim_config} already exists, skipping...")
        return

    # Use sim_hash to ensure reproducibility, but have variance in seed
    sim_hash = hashlib.sha256(sim_config.encode('utf-8')).digest()
    seed = int.from_bytes(sim_hash[:4], byteorder='little')

    sim = DextramerSimulator()
    mdata = sim.simulate_pmhc_data_from_distribution(total_cells=params.total_cells,
                                                     nof_clones=params.nof_clones,
                                                     p_binding_outlier=params.p_binding_outlier,
                                                     binding_ratio=params.binding_ratio,
                                                     mean_inc=params.mean_inc,
                                                     simulate_neg_control=True,
                                                     plot_data=False,
                                                     rep=params.i,
                                                     rng_key=seed,
                                                     )

    mdata.write(params.f_out)

    print(f"Finished simulations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--f_out", type=str, default="benchmarks/scenario_test/simulation/test.h5mu",
                        help="Output file path")
    parser.add_argument("--total_cells", type=int, default=5000, help="Number of cells")
    parser.add_argument("--nof_clones", type=int, default=None, help="Number of clones")
    parser.add_argument("--frac_clones", type=float, default=0.4,
                        help="Fraction of clones, only used when nof_clones is None")
    parser.add_argument("--p_binding_outlier", type=float, default=0.0,
                        help="Probability that cell from binding clone has low UMI count", )
    parser.add_argument("--binding_ratio", type=float, help="Ratio of binder cells", default=0.1)
    parser.add_argument("--mean_inc", type=float, help="Signal to noise ratio", default=20)
    parser.add_argument("--i", type=int, help="Repetition index", default=0)

    params = parser.parse_args()
    params = convert_str_to_bool_and_none(params)

    main(params)
