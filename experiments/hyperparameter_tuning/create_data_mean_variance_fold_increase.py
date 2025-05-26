import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
from dextrademixer.utils import DextramerSimulator
import hashlib


def main():
    output_h5mu = sys.argv[1]
    output_pdf = sys.argv[2]
    total_cell = int(sys.argv[3])
    nclones = int(sys.argv[4])
    p_outlier = float(sys.argv[5])
    p = float(sys.argv[6])
    cov = bool(int(sys.argv[7]))
    mean_inc = float(sys.argv[8])
    var_inc = float(sys.argv[9])
    i = int(sys.argv[10])

    sim_config = f'{total_cell}_{nclones}_{p_outlier}_{p}_{cov}_{mean_inc}_{var_inc}_{i}'
    # Use sim_hash to ensure reproducibility, but have variance in seed
    sim_hash = hashlib.sha256(sim_config.encode('utf-8')).digest()
    seed = int.from_bytes(sim_hash[:4], byteorder='little')

    plt.ioff()
    sim = DextramerSimulator()

    mdata1, fig = sim.simulate_pmhc_data_from_distribution(total_cells=total_cell,
                                                            nof_clones=nclones,
                                                            p_binding_outlier=p_outlier,
                                                            binding_ratio=p,
                                                            binding_fold_increase_range=[mean_inc],
                                                            variance_fold_increase_range=[var_inc],
                                                            simulate_neg_control=True,
                                                            use_clonotype_cov=cov,
                                                            plot_data=True,
                                                            rng_key=seed)
    mdata1.write(output_h5mu)
    fig.savefig(output_pdf)
    plt.close()


if __name__ == "__main__":
    main()
