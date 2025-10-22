import argparse
import os
import warnings

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import muon as mu

from tqdm import tqdm

import sys
sys.path.append("../../")
from dextrademixer.model import BEAMT
from dextrademixer.utils import calculate_metrics


def main(args):
    input_dir = os.path.join('benchmarks', args.scenario, 'simulation')
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5mu')]
    base_dir = os.path.join('benchmarks', args.scenario)

    def run_beamt(f_in):
        data_config = os.path.basename(f_in).replace('.h5mu', '')
        config = "BEAMT" + '_' + data_config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdata = mu.read(f_in)

        y_true = mdata.mod["airr"].obs["is_binder"]

        mixer = BEAMT()
        mixer.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control")
        mixer.fit()

        results = []
        posterior_params = [(0.01, None), (0.02, None), (0.05, None), (0.1, None), (0.2, None), (None, 0.5)]

        for target_fdr, threshold in posterior_params:
            p_pred, assignment = mixer.predict_posterior_class(target_fdr=target_fdr, threshold=threshold)
            results_dict = {}
            results_dict['config'] = config
            results_dict['model_config'] = "BEAMT"
            results_dict['data_config'] = data_config
            results_dict.update({'sim_' + k: v for k, v in mdata['gex'].uns['sim_params'].items()})
            results_dict.update({'posterior_target_fdr': target_fdr, 'posterior_threshold': threshold,
                                 "posterior_config": f"{target_fdr}_{threshold}"})

            results_dict.update(calculate_metrics(y_true, p_pred, assignment))
            results.append(results_dict)

        results = pd.DataFrame(results)
        csv_dir = os.path.join(base_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        results.to_csv(csv_dir + f"/{config}.csv")

    with ThreadPoolExecutor() as ex:
        list(tqdm(ex.map(run_beamt, input_files), total=len(input_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tool on all data from a scenario.")
    # IO parameters
    parser.add_argument("--scenario", type=str, default="scenario_test", help="Name of scenario")
    args = parser.parse_args()
    main(args)
