import argparse
import multiprocessing
import os
import warnings
import pickle
from time import strftime

import sys


sys.path.append("../../")

import numpyro
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score)

from dextrademixer.model import DextraDemixer
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu

multiprocessing.set_start_method('spawn', force=True)


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
    parser.add_argument("--maxiter", type=int, default=5000, help="Upper limit for maxiter for optuna")
    return parser.parse_args()


def run_inference(opt_params, f_in,  model_type, m, neg_ctrl, ir_clone, threads, seed, trial_number):
    numpyro.set_host_device_count(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)
    y_true = mdata.mod["airr"].obs["is_binder"]

    mixer = DextraDemixer(model_type=model_type, mode=m)
    mixer.preprocess_model_data(mdata, "pmhc1",
                                neg_ctrl_key=neg_ctrl,
                                ir_clone_key=ir_clone)

    trace, best_loss = mixer.fit_svi(svi_config=opt_params,
                                     nof_inits=opt_params["nof_inits"],
                                     rng_key=seed,
                                     return_loss=True)
    p_pred, assignment_fdr = mixer.predict_posterior_class(target_fdr=0.05)
    config = f"{model_type}_{m}_{neg_ctrl}_{ir_clone}_{f_in.replace('simulation/sim_', '').replace('.h5mu', '')}_Trial={trial_number}"
    os.makedirs('saved_models', exist_ok=True)
    with open(f"saved_models/{config}.pkl", "wb") as f:
        pickle.dump(mixer, f)

    return y_true, p_pred, assignment_fdr, best_loss


def worker(dataset, opt_params, model_type, mode, neg, clonotype, threads, seed, trial_number):
    y_true, p_pred, assignment_fdr, best_loss = run_inference(opt_params, dataset, model_type, mode, neg, clonotype,
                                                              threads, seed, trial_number)
    return y_true, p_pred, assignment_fdr, best_loss


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    def objective(trial):
        """Optuna objective function."""
        init_value = trial.suggest_float("init_value", 1e-4, 1e0, log=True)
        # max_iter = trial.suggest_int("maxiter", 100, args.maxiter, log=True)
        # transition_steps = trial.suggest_float("transition_rate", 0.0, 1.0) * max_iter

        opt_params = {"maxiter": args.max_iter,
                      "nof_inits": 10,
                       "adam":
                               {
                                "init_value": init_value,
                                # "transition_steps": transition_steps,
                                # "decay_rate": trial.suggest_float("decay_rate", 0.5, 1.0, log=True),
                                # "end_value": trial.suggest_float("end_value_factor", 1e-3, 1e0, log=True) * init_value,
                               },
                      # "adam_beta":
                      #           {
                      #            "b1": trial.suggest_float("beta1", 0.5, 1.0, log=True),
                      #            "b2": trial.suggest_float("beta2", 0.5, 1.0, log=True),
                      #           }
                      }

        # Evaluate over multiple datasets
        with multiprocessing.Pool(processes=args.threads) as pool:
             results = pool.starmap(worker, [(dataset, opt_params, args.model_type, args.mode,
                                            args.neg, args.clonotype, args.threads, args.seed, trial.number)
                                            for dataset in args.input_files])
        y_true, p_pred, assignment_fdr, best_loss = zip(*results)

        f1_list = []
        for i, dataset in enumerate(args.input_files):
            if not np.isfinite(p_pred[i]).any() or not np.isfinite(assignment_fdr[i]).any():
                f1 = 0.0
                for metric in ["f1", "precision", "recall", "aps", "acc"]:
                    trial.set_user_attr(f"{metric}_{dataset}", 0.0)
            else:
                trial.set_user_attr(f"auc_{dataset}", roc_auc_score(y_true[i], p_pred[i]))
                f1 = f1_score(y_true[i], assignment_fdr[i])
                trial.set_user_attr(f"f1_{dataset}", f1)
                trial.set_user_attr(f"precision_{dataset}", precision_score(y_true[i], assignment_fdr[i]))
                trial.set_user_attr(f"recall_{dataset}", recall_score(y_true[i], assignment_fdr[i]))
                trial.set_user_attr(f"aps_{dataset}", average_precision_score(y_true[i], p_pred[i]))
                trial.set_user_attr(f"acc_{dataset}", accuracy_score(y_true[i], assignment_fdr[i]))
            f1_list.append(f1)
            trial.set_user_attr(f"converged_{dataset}", best_loss[i]["converged"])
            trial.set_user_attr(f"best_loss_{dataset}", float(best_loss[i]["best_loss"]))
            trial.set_user_attr(f"init_loss_{dataset}", float(best_loss[i]["init_loss"]))
            trial.set_user_attr(f"best_iteration_{dataset}", int(best_loss[i]["best_iteration"]))

        mean_f1 = np.mean(f1_list)

        return mean_f1

    sampler = optuna.samplers.GPSampler(seed=args.seed)
    study_name = f"{strftime('%Y%m%d-%H%M%S')}_{args.model_type}_mode{args.mode}_neg{args.neg}_clonotype{args.clonotype}"
    os.makedirs('optuna_study', exist_ok=True)

    study = optuna.create_study(storage=f"sqlite:///optuna_study/{study_name}.db",
                                sampler=sampler, direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=100)

    df = study.trials_dataframe()
    df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
