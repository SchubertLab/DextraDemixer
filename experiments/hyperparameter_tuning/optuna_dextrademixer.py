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
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on multiple simulation files and average results.")
    parser.add_argument("input_files", nargs="+", help="List of input .h5mu files")
    parser.add_argument("output_file", help="Output CSV file for averaged results")
    parser.add_argument("--mode", required=True, help="Processing mode")
    parser.add_argument("--neg", required=True, help="Negative control parameter")
    parser.add_argument("--clonotype", required=True, help="Clonotype parameter")
    parser.add_argument("--model_type", required=True, help="Model type",
                        choices=["mixturemodelkmeans", "mixturemodel"])
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def run_inference(opt_params, f_in,  model_type, m, neg_ctrl, ir_clone, threads, seed):
    if threads is not None:
        numpyro.set_host_device_count(threads)

    mdata = mu.read(f_in)
    y_true = mdata.mod["airr"].obs["is_binder"]

    mixer = DextraDemixer(model_type=model_type, mode=m)
    mixer.preprocess_model_data(mdata, "pmhc1",
                                neg_ctrl_key=neg_ctrl,
                                ir_clone_key=ir_clone)

    trace = mixer.fit_svi(svi_config=opt_params, rng_key=seed)
    p_pred, assignment_fdr = mixer.predict_posterior_class(target_fdr=0.05)
    auc = roc_auc_score(y_true, p_pred, average="weighted")

    return auc


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    def objective(trial):
        """Optuna objective function."""

        opt_params = {"maxiter": trial.suggest_int("maxiter", 100, 10000, log=True),
                       "adam":
                               {
                                "init_value": trial.suggest_float("init_value", 1e-5, 1e-1, log=True),
                                "transition_steps": trial.suggest_int("transition_steps", 100, 5000),
                                "decay_rate": trial.suggest_float("decay_rate", 0.5, 1.0),
                                "end_value": trial.suggest_float("end_value", 1e-8, 1e-2, log=True)
                               }
                      }

        # Evaluate over multiple datasets
        mean_auc = np.mean([run_inference(opt_params, dataset, args.model_type, args.mode,
                                          args.neg, args.clonotype, args.threads, args.seed)
                            for dataset in args.input_files])

        return mean_auc

    sampler = optuna.samplers.GPSampler(args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()
    df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
