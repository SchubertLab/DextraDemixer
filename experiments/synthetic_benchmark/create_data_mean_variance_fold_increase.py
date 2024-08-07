import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
from dextramixer.utils import DextramerSimulator


def main():
    output_h5mu = sys.argv[1]
    output_pdf = sys.argv[2]
    total_cell = int(sys.argv[3])
    nclones = int(sys.argv[4])
    p_outlier = float(sys.argv[5])
    p = float(sys.argv[6])
    cov = bool(sys.argv[7])
    mean_inc = float(sys.argv[8])
    var_inc = float(sys.argv[9])
    n = int(sys.argv[10])

    plt.ioff()
    sim = DextramerSimulator()
    for i in range(0, n):
        mdata1, axs1 = sim.simulate_pmhc_data_from_distribution(total_cells=total_cell,
                                                                nof_clones=nclones,
                                                                p_binding_outlier=p_outlier,
                                                                binding_ratio=p,
                                                                binding_fold_increase_range=[mean_inc],
                                                                variance_fold_increase_range=[var_inc],
                                                                simulate_neg_control=True,
                                                                use_clonotype_cov=cov,
                                                                plot_data=True,
                                                                rng_key=i)
        mdata1.write(output_h5mu)
        axs1[0, 0].get_figure().savefig(output_pdf)
        plt.close()


if __name__ == "__main__":
    main()
