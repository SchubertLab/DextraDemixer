import argparse
import multiprocessing
import os
import warnings
import pickle
from time import strftime

import numpyro
import optuna
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score)

import sys


sys.path.append("../../")
from dextrademixer.model import DextraDemixer
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu

multiprocessing.set_start_method("spawn", force=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on multiple simulation files and average results.")
    parser.add_argument("input_files", nargs="+", help="List of input .h5mu files")
    parser.add_argument("output_file", help="Output CSV file for averaged results")
    parser.add_argument("--mode", required=True, help="Processing mode")
    parser.add_argument("--neg", required=True, help="Negative control parameter")
    parser.add_argument("--clonotype", required=True, help="Clonotype parameter")
    parser.add_argument("--model_type", default='mixturemodelkmeans', help="Model type",
                        choices=["mixturemodelkmeans", "mixturemodel"])
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--maxiter", type=int, default=5000, help="Upper limit for maxiter for optuna")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials, in this case number of seeds")
    parser.add_argument("--mp", type=str, default=True, help="Use multiprocessing")
    parser.add_argument("--alpha_model", type=str, default='overdispersion',
                        choices=["overdispersion", "kmeans"],
                        help="Modeling of the alpha parameter. Options: 'overdispersion', 'kmeans'.")
    parser.add_argument("--overdispersion_scale_prior", type=float, default=1,
                        help="Prior for scale parameter of HalfCauchy for overdispersion model. Not used for kmeans.")
    parser.add_argument("--var_hyperprior", type=float, default=10,
                        help="Prior for scale parameter of kmeans model. Not used for overdispersion.")
    return parser.parse_args()


def run_inference(opt_params, f_in, model_type, m, alpha_model, neg_ctrl, ir_clone, seed, trial_number):
    numpyro.set_host_device_count(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)
    y_true = mdata.mod["airr"].obs["is_binder"]

    mixer = DextraDemixer(model_type=model_type, mode=m, alpha_model=alpha_model)
    mixer.preprocess_model_data(mdata, "pmhc1",
                                neg_ctrl_key=neg_ctrl,
                                ir_clone_key=ir_clone)

    mixer.model._model_config["overdispersion_scale_prior"] = opt_params["overdispersion_scale_prior"]
    mixer.model._model_config["var_hyperprior"] = opt_params["var_hyperprior"]

    trace, best_loss = mixer.fit_svi(svi_config=opt_params,
                                     nof_inits=opt_params["nof_inits"],
                                     rng_key=seed,
                                     return_loss=True)
    p_pred, assignment_fdr = mixer.predict_posterior_class(threshold=0.5)

    config = (f"{model_type}_{m}_{neg_ctrl}_{ir_clone}_{alpha_model}_{opt_params['overdispersion_scale_prior']},{opt_params['var_hyperprior']}_lr={opt_params['adam']['init_value']}"
              f"{f_in.replace('simulation/sim_', '').replace('.h5mu', '')}"
              f"_Trial={trial_number}")

    mixer.plot_results(assignment_fdr, p_pred, y_true, seed, config)

    os.makedirs("saved_models", exist_ok=True)
    with open(f"saved_models/{config}.pkl", "wb") as f:
        pickle.dump(mixer, f)

    return y_true, p_pred, assignment_fdr, best_loss


def worker(dataset, opt_params, model_type, mode, alpha_model, neg, clonotype, seed, trial_number):
    try:
        y_true, p_pred, assignment_fdr, best_loss = run_inference(opt_params, dataset, model_type, mode,
                                                                  alpha_model, neg, clonotype, seed, trial_number)
    except Exception as e:
        raise RuntimeError(f"Failed on {dataset}") from e

    return y_true, p_pred, assignment_fdr, best_loss


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    def objective(trial):
        """Optuna objective function."""
        init_value = trial.suggest_float("init_value", 1e-4, 1e0, log=True) if args.lr is None else args.lr

        opt_params = {"maxiter": args.maxiter,
                      "nof_inits": 10,
                      "adam":
                          {
                              "init_value": init_value,
                          },
                      "overdispersion_scale_prior": args.overdispersion_scale_prior,
                      "var_hyperprior": args.var_hyperprior,
                      }

        # Evaluate over multiple datasets
        if args.mp:
            with multiprocessing.Pool(processes=args.threads) as pool:
                results = pool.starmap(worker, [(dataset, opt_params, args.model_type, args.mode,
                                                 args.alpha_model, args.neg, args.clonotype, args.seed, trial.number)
                                                for dataset in args.input_files], chunksize=1)
        else:
            results = []
            for dataset in args.input_files:
                results.append(worker(dataset, opt_params, args.model_type, args.mode,
                                      args.alpha_model, args.neg, args.clonotype, args.seed, trial.number))

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
    study_name = f"{strftime('%Y%m%d-%H%M%S')}_mode{args.mode}_neg{args.neg}_clonotype{args.clonotype}_lr{args.lr}_alpha_model{args.alpha_model}_priors{args.overdispersion_scale_prior},{args.var_hyperprior}"
    os.makedirs("optuna_study", exist_ok=True)

    study = optuna.create_study(storage=f"sqlite:///optuna_study/{study_name}.db",
                                sampler=sampler, direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=args.n_trials)

    df = study.trials_dataframe()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
