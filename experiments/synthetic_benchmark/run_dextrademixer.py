import argparse
import multiprocessing
import os
import warnings
import pickle
from time import strftime
import pandas as pd
import numpyro
import optuna
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, classification_report)
import sys

sys.path.append("../../")
from dextrademixer.model import DextraDemixer
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu

multiprocessing.set_start_method("spawn", force=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on single file.")
    # Data parameters
    parser.add_argument("--input_file", type=str, default="simulation/test.h5mu", help="Input .h5mu file")
    parser.add_argument("--output_file", type=str, default="output.csv",
                        help="Output CSV file for averaged results")
    parser.add_argument("--pmhc_key", type=str, default="pmhc1", help="Processing mode")
    parser.add_argument("--gex_key", type=str, default="gex", help="Key for multimer counts")
    parser.add_argument("--label_key", type=str, default="is_binder", help="Key for labels")

    # Model parameters
    parser.add_argument("--model_type", default='mixturemodelkmeans', help="Model type",
                        choices=["mixturemodelkmeans", "mixturemodel"])
    parser.add_argument("--mode", type=str, default="I", help="Processing mode")
    parser.add_argument("--neg_ctrl_key", type=str, default=None, help="Negative control parameter")
    parser.add_argument("--ir_clone_key", type=str, default='clone_id', help="Clonotype parameter")
    parser.add_argument("--alpha_model", type=str, default="kmeans",
                        choices=["overdispersion", "kmeans"],
                        help="Modeling of the alpha parameter. Options: 'overdispersion', 'kmeans'.")
    parser.add_argument("--overdispersion_scale_prior", type=float, default=1e-2,
                        help="Prior for scale parameter of HalfCauchy for overdispersion model. Not used for kmeans.")
    parser.add_argument("--var_hyperprior", type=float, default=1e0,
                        help="Prior for scale parameter of kmeans model. Not used for overdispersion.")
    parser.add_argument("--outlier_threshold", type=float, default=4.0,
                        help="Threshold for outlier removal based on log transformed z-score.")
    parser.add_argument("--target_fdr", type=float, default=0.05,
                        help="Target FDR for posterior class assignment. If None, uses threshold instead.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for posterior class assignment. If None, uses target_fdr instead.")

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--maxiter", type=int, default=1000, help="Upper limit for maxiter for optuna")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    return parser.parse_args()


def run_inference(f_in, args):
    opt_params = {"maxiter": args.maxiter,
                  "nof_inits": 10,
                  "adam": {"init_value": args.lr,},
                  "overdispersion_scale_prior": args.overdispersion_scale_prior,
                  "var_hyperprior": args.var_hyperprior,
                  }

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

    mixer.model._model_config["overdispersion_scale_prior"] = opt_params["overdispersion_scale_prior"]
    mixer.model._model_config["var_hyperprior"] = opt_params["var_hyperprior"]

    trace, best_loss = mixer.fit_svi(svi_config=opt_params,
                                     nof_inits=opt_params["nof_inits"],
                                     rng_key=args.seed,
                                     return_loss=True)
    p_pred, assignment = mixer.predict_posterior_class(target_fdr=args.target_fdr, threshold=args.threshold)

    config = (f"{args.model_type}_{args.mode}_{args.neg_ctrl_key}_{args.ir_clone_key}_{args.alpha_model}_{opt_params['overdispersion_scale_prior']},{opt_params['var_hyperprior']}_lr={opt_params['adam']['init_value']}\n"
              f"{f_in.replace('simulation/sim_', '').replace('.h5mu', '').replace('/', '-')}")

    print(classification_report(y_true[y_true.notna()].astype(int), assignment[y_true.notna().values]))
    mixer.plot_results(assignment, p_pred, y_true, args.seed, config)

    os.makedirs("saved_models", exist_ok=True)
    with open(f"saved_models/{config}.pkl", "wb") as f:
        pickle.dump(mixer, f)

    return y_true, p_pred, assignment, best_loss


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    y_true, p_pred, assignment, best_loss = run_inference(args.input_file, args)

    results = pd.Series()
    results['roc_auc'] = roc_auc_score(y_true[y_true.notna()].astype(int), p_pred[y_true.notna()])
    results['pr_auc'] = average_precision_score(y_true[y_true.notna()].astype(int), p_pred[y_true.notna()])
    results['f1'] = f1_score(y_true[y_true.notna()].astype(int), assignment[y_true.notna()])
    results['precision'] = precision_score(y_true[y_true.notna()].astype(int), assignment[y_true.notna()])
    results['recall'] = recall_score(y_true[y_true.notna()].astype(int), assignment[y_true.notna()])
    results['accuracy'] = accuracy_score(y_true[y_true.notna()].astype(int), assignment[y_true.notna()])
    print(results)
    results.to_csv(args.output_file)


if __name__ == "__main__":
    main()
