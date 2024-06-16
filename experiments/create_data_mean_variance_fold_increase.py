import sys
sys.path.append("../")

import matplotlib.pyplot as plt
from dextramixer.utils import DextramerSimulator


def main():

    total_cell = int(sys.argv[1])
    nclones = int(sys.argv[2])
    p = float(sys.argv[3])
    cov = int(sys.argv[4])
    mean_inc = float(sys.argv[5])
    var_inc = float(sys.argv[6])
    n = int(sys.argv[7])

    plt.ioff()
    sim = DextramerSimulator()
    for i in range(0, n):
        f = f"simulation/sim_ncell{total_cell}_nclone{nclones}_pbinder{p}_negctr1_cov{cov}_meaninc{mean_inc}_varinc{var_inc}_rep{i}"
        mdata1, axs1 = sim.simulate_pmhc_data_from_distribution(total_cells=total_cell,
                                                                nof_clones=nclones,
                                                                binding_ratio=p,
                                                                binding_fold_increase_range=[mean_inc],
                                                                variance_fold_increase_range=[var_inc],
                                                                simulate_neg_control=True,
                                                                use_clonotype_cov=cov,
                                                                plot_data=True,
                                                                rng_key=i)
        mdata1.write(f + ".h5mu")
        axs1[0, 0].get_figure().savefig(f + ".pdf")
        plt.close()


if __name__ == "__main__":
    main()
