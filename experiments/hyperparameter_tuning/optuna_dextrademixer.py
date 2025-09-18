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
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import sys

sys.path.append("../../")
from dextrademixer.model import DextraDemixer
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu

multiprocessing.set_start_method("spawn", force=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on multiple simulation files and average results.")
    # Data parameters
    parser.add_argument("input_files", nargs="+", help="List of input .h5mu files")
    parser.add_argument("output_file", help="Output CSV file for averaged results")
    parser.add_argument("--pmhc_key", type=str, default="pmhc1", help="Key for pMHC counts")
    parser.add_argument("--gex_key", type=str, default="gex", help="Key for pMHC count modality. "
                                                                   "Usually saved in 'gex' modality.")
    parser.add_argument("--label_key", type=str, default="is_binder", help="Key for binder labels")

    # Model parameters
    parser.add_argument("--model_type", default='mixturemodelkmeans', help="Model type",
                        choices=["mixturemodelkmeans", "mixturemodel"])
    parser.add_argument("--mode", type=str, default="C", help="Processing mode")
    parser.add_argument("--neg_ctrl_key", type=str, default=None, help="Negative control key. "
                                                                       "If none, no negative control is used.")
    parser.add_argument("--ir_clone_key", type=str, default='clone_id', help="Clonotype id key. "
                                                                             "If None, no clonotype id is used.")
    parser.add_argument("--alpha_model", type=str, default="overdispersion",
                        choices=["overdispersion", "kmeans"],
                        help="Modeling of the alpha parameter. Options: 'overdispersion', 'kmeans'.")
    parser.add_argument("--overdispersion_scale_prior", type=float, default=None,
                        help="Prior for scale parameter of HalfCauchy for overdispersion model. "
                             "Not used if alpha_model == kmeans. None for optuna to suggest.")
    parser.add_argument("--prior_value", type=float, default=None,
                        help="Prior for scale parameter if alpha_model == kmeans "
                             "or scale for HalfCauchy if alpha_model == overdispersion. "
                             "None for optuna to suggest.")
    parser.add_argument("--outlier_threshold", type=float, default=4.0,
                        help="Threshold for outlier removal based on log transformed z-score.")
    parser.add_argument("--target_fdr", type=float, default=0.05,
                        help="Target FDR for posterior class assignment. If None, uses threshold instead.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for posterior class assignment. If None, uses target_fdr instead.")

    # Optimization parameters
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_inits", type=int, default=10, help="Number of initializations for SVI."
                                                                "Can be set low, if using k-means to initialize.")
    parser.add_argument("--maxiter", type=int, default=10000,
                        help="Maximum number of iterations for optimization."
                             "If None, uses optuna to suggest.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for Adam optimizer. If None, uses optuna to suggest.")
    parser.add_argument("--mp", type=str, default=True, help="Use multiprocessing")
    return parser.parse_args()


def run_inference(f_in: str, args: argparse.Namespace, opt_params: dict, trial_number: int=-1):
    numpyro.set_host_device_count(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)
    y_true = mdata.mod["airr"].obs[args.label_key]

    mixer = DextraDemixer(model_type=args.model_type, mode=args.mode, alpha_model=args.alpha_model)
    mixer.preprocess_model_data(mdata,
                                pmhc_key=args.pmhc_key,
                                gex_key=args.gex_key,
                                neg_ctrl_key=args.neg_ctrl_key,
                                ir_clone_key=args.ir_clone_key,
                                outlier_threshold=args.outlier_threshold,)

    mixer.model._model_config["overdispersion_scale_prior"] = opt_params["prior_value"]
    mixer.model._model_config["var_hyperprior"] = opt_params["prior_value"]

    trace, best_loss = mixer.fit_svi(svi_config=opt_params,
                                     nof_inits=opt_params["n_inits"],
                                     rng_key=args.seed,
                                     return_loss=True)
    p_pred, assignment = mixer.predict_posterior_class(target_fdr=args.target_fdr, threshold=args.threshold)

    config = (f"{args.model_type}_{args.mode}_neg={args.neg_ctrl_key}_clone={args.ir_clone_key}_{args.alpha_model}_"
              f"prior={opt_params['prior_value']}_"
              f"lr{opt_params['adam']['init_value']}_"
              f"{f_in.replace('simulation/sim_', '').replace('.h5mu', '')}_"
              f"_Trial={trial_number}")
    config = config.replace("/", "_").replace(":", "_")

    mixer.plot_results(assignment, p_pred, y_true, args.seed, config)

    os.makedirs("saved_models", exist_ok=True)
    with open(f"saved_models/{config}.pkl", "wb") as f:
        pickle.dump(mixer, f)

    return y_true, p_pred, assignment, best_loss


def worker(f_in, args, opt_params, trial_number=-1):
    # If multiprocessing is enabled, the following is needed to track which dataset failed
    if args.mp:
        try:
            y_true, p_pred, assignment_fdr, best_loss = run_inference(f_in, args, opt_params, trial_number=trial_number)
        except Exception as e:
            raise RuntimeError(f"Failed on {f_in}") from e
    else:
        y_true, p_pred, assignment_fdr, best_loss = run_inference(f_in, args, opt_params, trial_number=trial_number)
    return y_true, p_pred, assignment_fdr, best_loss


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # Evaluate over multiple datasets
        opt_params = {"maxiter": trial.suggest_int("maxiter", 500, 50000, log=True)
                                 if args.maxiter is None else args.maxiter,
                      "n_inits": args.n_inits,
                      "adam": {"init_value": trial.suggest_float("init_value", 1e-4, 1e0, log=True)
                                             if args.lr is None else args.lr, },
                      "prior_value": trial.suggest_float("prior_value", 1e-2, 1e1, log=True)
                                     if args.prior_value is None else args.prior_value,
                      }
        if args.mp:
            with multiprocessing.Pool(processes=args.threads) as pool:
                results = pool.starmap(worker, [(f_in, args, opt_params, trial.number)
                                                for f_in in args.input_files], chunksize=1)
        else:
            results = []
            for f_in in args.input_files:
                results.append(worker(f_in, args, opt_params, trial.number))

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
    study_name = (f"{strftime('%Y%m%d-%H%M%S')}_mode={args.mode}_neg={args.neg_ctrl_key}_clone={args.ir_clone_key}_"
                  f"lr={args.lr}_alpha_model={args.alpha_model}_"
                  f"prior={args.prior_value}")
    os.makedirs("optuna_study", exist_ok=True)

    storage = JournalStorage(JournalFileBackend(f"optuna_study/{study_name}.log"))
    study = optuna.create_study(storage=storage,
                                sampler=sampler, direction="maximize", study_name=study_name,)
    study.optimize(objective, n_trials=args.n_trials)

    df = study.trials_dataframe()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
