import argparse
import os
import warnings

import pandas as pd
import muon as mu

import sys
sys.path.append("../../")
from dextrademixer.model import BEAMT
from dextrademixer.utils import calculate_metrics


def run_inference(args):
    f_in = args.f_in
    f_out = args.f_out

    base_dir = os.path.dirname(f_out)
    sim_config = os.path.basename(f_in).replace('.h5mu', '')
    config = "BEAM" + '_' + sim_config
    csv_dir = os.path.join(base_dir, 'csv')
    if os.path.exists(csv_dir + f"/{config}.csv"):
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)

    y_true = mdata.mod["airr"].obs["is_binder"]

    model = BEAMT()
    model.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control")
    model.fit()

    p_pred, assignment = model.predict_posterior_class(threshold=0.5)
    results_dict = {'config': config, 'model_config': "BEAM", 'sim_config': sim_config, 'threshold': 0.5}
    results_dict.update(calculate_metrics(y_true, p_pred, assignment, full_metrics=False))
    results_dict.update({'sim_'+k: mdata['gex'].uns['sim_params'][k] for k in ['binding_ratio', 'mean_inc', 'nof_clones', 'p_binding_outlier', 'rep', 'rng_key', 'total_cells']})

    results = pd.DataFrame([results_dict])
    os.makedirs(csv_dir, exist_ok=True)
    results.to_csv(f_out)


def main():
    parser = argparse.ArgumentParser(description="Run tool on a single file.")
    parser.add_argument("--f_in", type=str, default="benchmarks/scenario_test/simulation/2000_800_0.0_0.9_False_500_None_1.h5mu",
                        help="Path to input h5mu file")
    parser.add_argument("--f_out", type=str, default="benchmarks/scenario_test/csv/2000_800_0.0_0.9_False_500_None_1.csv",
                        help="Path to output csv file")
    args = parser.parse_args()
    
    run_inference(args)
    print("DONE!")


if __name__ == "__main__":
    main()
