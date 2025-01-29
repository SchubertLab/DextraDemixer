import argparse

import sys


sys.path.append("../../")

import numpyro
import os
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dextrademixer.model import DextraDemixer
import muon as mu


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on multiple simulation files and average results.")
    parser.add_argument("input_files", nargs="+", help="List of input .h5mu files")
    parser.add_argument("output_file", help="Output CSV file for averaged results")
    parser.add_argument("--mode", required=True, help="Processing mode")
    parser.add_argument("--neg", required=True, help="Negative control parameter")
    parser.add_argument("--clonotype", required=True, help="Clonotype parameter")
    return parser.parse_args()


def run_inference(opt_params, f_in,  m, neg_ctrl, ir_clone):
    numpyro.set_host_device_count(4)

    mdata = mu.read(f_in)
    y_true = mdata.mod["airr"].obs["is_binder"]

    mixer = DextraDemixer(model_type="mixturemodelkmeans", mode=m)
    mixer.preprocess_model_data(mdata, "pmhc1",
                                neg_ctrl_key=neg_ctrl,
                                ir_clone_key=ir_clone)

    trace = mixer.fit_svi(svi_config=opt_params)
    p_pred, assignment_fdr = mixer.predict_posterior_class(target_fdr=0.05)
    auc = roc_auc_score(y_true, p_pred, average="weighted")

    return auc


def main():
    args = parse_arguments()

    def objective(trial):
        """Optuna objective function."""

        opt_params = {"maxiter": trial.suggest_int("maxiter", 100, 10000),
                       "adam":
                               {
                                "init_value": trial.suggest_loguniform("init_value", 1e-5, 1e-1),
                                "transition_steps": trial.suggest_int("transition_steps", 100, 5000),
                                "decay_rate": trial.suggest_uniform("decay_rate", 0.5, 1.0),
                                "end_value": trial.suggest_loguniform("end_value", 1e-8, 1e-2)
                               }
                      }

        # Evaluate over multiple datasets
        mean_auc = np.mean([run_inference(opt_params, dataset, args.mode, args.neg, args.clonotype)
                            for dataset in args.input_files])

        return mean_auc

    sampler = optuna.samplers.GPSampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()
    df.write_csv(args.output_file)


if __name__ == "__main__":
    main()
