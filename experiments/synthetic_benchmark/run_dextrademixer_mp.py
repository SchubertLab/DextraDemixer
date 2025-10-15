import argparse
import multiprocessing
import os
import warnings
import pickle

import pandas as pd
import numpyro
import muon as mu

from tqdm import tqdm

import sys
sys.path.append("../../")
from dextrademixer.model import DextraDemixer
from dextrademixer.utils import (convert_str_to_bool_and_none, get_slurm_cpu_count, calculate_metrics,
                                 guess_worker_mem_limit_mb, init_worker)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on all data from a scenario.")
    # IO parameters
    parser.add_argument("--scenario", type=str, default="scenario_test",
                        help="Name of scenario")
    parser.add_argument("--use_mp", type=str, default=True,
                        help="Whether to use multiprocessing")

    # Data parameters
    parser.add_argument("--gex_key", type=str, default="gex",
                        help="Key for modality where pMHC counts are stored")
    parser.add_argument("--airr_key", type=str, default='airr',
                        help="Key for modality where TCR data is stored")
    parser.add_argument("--pmhc_key", type=str, default="pmhc1",
                        help="Key for pMHC counts, expected to be in 'gex' modality")
    parser.add_argument("--label_key", type=str, default="is_binder",
                        help="Key for labels, expected to be in 'airr' modality")

    # Model parameters
    parser.add_argument("--model_type", default='mixturemodelkmeans', help="Model type",
                        choices=["mixturemodelkmeans", "mixturemodel"])
    parser.add_argument("--mode", type=str, default="C",
                        help="Processing mode. 'I' -> independent concentration parameter for signal and noise data. "
                             "'C' -> clonotype level concentration parameter", choices=["I", "C"])
    parser.add_argument("--neg_ctrl_key", type=str, default=None,
                        help="Key for negative control counts. If not None, enables model to use negative control data for "
                             "fitting. Expected to be in 'gex' modality. If None, no negative control data is used.")
    parser.add_argument("--ir_clone_key", type=str, default='clone_id',
                        help="Key for clonotype information. If not None, enables model to have different "
                             "mixture coefficients for each clonotype. Expected to be in 'airr' modality. ")
    parser.add_argument("--alpha_model", type=str, default="kmeans",
                        choices=["overdispersion", "kmeans"],
                        help="Modeling of the alpha parameter. Options: 'overdispersion', 'kmeans'.")
    parser.add_argument("--hyperprior", type=float, default=1e0,
                        help="Prior for scale parameter of kmeans model and HalfCauchy for overdispersion model")
    parser.add_argument("--outlier_threshold", type=float, default=4.0,
                        help="Threshold for outlier removal based on log transformed z-score.")

    # Posterior class assignment parameters
    # parser.add_argument("--target_fdr", type=float, default=None,
    #                     help="Target FDR for posterior class assignment. If None, uses threshold instead.")
    # parser.add_argument("--threshold", type=float, default=None,
    #                     help="Threshold for posterior class assignment. If None, uses target_fdr instead.")

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--maxiter", type=int, default=50, help="Number of iterations for optimization")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    return parser.parse_args()


def parse_data_config_from_name(filename):
    data_config_dict = {}
    parts = filename.replace('.h5mu', '').split('_')
    data_config_dict['total_cell'] = int(parts[0])
    data_config_dict['nof_clones'] = int(parts[1])
    data_config_dict['p_binding_outlier'] = float(parts[2])
    data_config_dict['binding_ratio'] = float(parts[3])
    data_config_dict['use_clonotype_cov'] = parts[4] == 'True'
    data_config_dict['mean_inc'] = int(parts[5])
    data_config_dict['var_inc'] = int(parts[6])
    data_config_dict['i'] = int(parts[7])
    return data_config_dict


def run_inference_star(t):
    """ Helper for multiprocessing; unpack args """
    f, args = t
    run_inference(f, args)


def run_inference(f_in, args):
    model_config = (f"{args.model_type}_{args.mode}_{args.neg_ctrl_key}_{args.ir_clone_key}_"
                    f"{args.alpha_model}_{args.hyperprior}_{args.lr}")
    data_config = os.path.basename(f_in).replace('.h5mu', '')
    config = model_config + '_' + data_config

    base_dir = os.path.join('benchmarks', args.scenario)
    if os.path.exists(os.path.join(base_dir, 'csv', f"{config}.csv")):
        print(f"Results for {config} already exist, skipping...")
        return

    numpyro.set_host_device_count(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)

    mixer = DextraDemixer(model_type=args.model_type, mode=args.mode, alpha_model=args.alpha_model)
    mixer.preprocess_model_data(mdata, pmhc_key=args.pmhc_key, gex_key=args.gex_key, neg_ctrl_key=args.neg_ctrl_key,
                                ir_clone_key=args.ir_clone_key, outlier_threshold=args.outlier_threshold, )

    mixer.model._model_config["overdispersion_scale_prior"] = args.hyperprior
    mixer.model._model_config["var_hyperprior"] = args.hyperprior
    opt_params = {"maxiter": args.maxiter, "nof_inits": 10, "adam": {"init_value": args.lr, }, }
    mixer.fit_svi(svi_config=opt_params, nof_inits=opt_params["nof_inits"], rng_key=args.seed, return_loss=True)

    # Save model
    save_model_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_model_dir, exist_ok=True)
    mixer.save_model(os.path.join(save_model_dir, f"{config}.pkl"))

    # Log metrics
    y_true = mdata.mod[args.airr_key].obs[args.label_key].astype(int).values
    posterior_samples = mixer.get_posterior_samples(num_samples=1000, seed=args.seed)

    results = []
    # Predict posterior class with different methods, (target_fdr, threshold, quantile, cred_intvl)
    posterior_params = [
        (0.01, None, None, None), (0.02, None, None, None), (0.05, None, None, None), (0.10, None, None, None),
        (0.20, None, None, None), (None, 0.50, None, None), (0.05, None, 0.50, None), (0.05, None, 0.40, None),
        (0.05, None, 0.30, None), (0.05, None, 0.20, None), (0.05, None, 0.10, None), (0.05, None, None, 0.50),
        (0.05, None, None, 0.60), (0.05, None, None, 0.70), (0.05, None, None, 0.80), (0.05, None, None, 0.90),]

    for target_fdr, threshold, quantile, cred_intvl in posterior_params:
        p_pred, assignment = mixer.predict_posterior_class(target_fdr=target_fdr, threshold=threshold,
                                                          quantile=quantile, cred_intvl=cred_intvl)
        posterior_config = f"{target_fdr}_{threshold}_{quantile}_{cred_intvl}"
        # Plot results
        os.makedirs(os.path.join(base_dir, 'figs', config), exist_ok=True)
        sim_params = mdata[args.gex_key].uns['sim_params']
        additional_text = (f"p_binder: {sim_params['binding_ratio']}\n"
                           f"q_noise: {sim_params['mean_non_binder']:.2f}, "
                           f"alpha_noise: {sim_params['concentration_non_binder']:.2f}\n"
                           f"q_signal: {sim_params['mean_pos']:.2f}, "
                           f"alpha_signal: {sim_params['concentration_pos']:.2f}")
        mixer.plot_results(assignment, p_pred, y_true if args.label_key is not None else None, args.seed,
                           config + '/' + posterior_config, additional_text=additional_text,
                           save_dir=os.path.join(base_dir, 'figs'), show=False)

        results_dict = {'config': config, 'model_config': model_config, 'data_config': data_config,
                        'posterior_target_fdr': target_fdr, 'posterior_threshold': threshold,
                        'posterior_quantile': quantile, 'posterior_cred_intvl': cred_intvl,
                        "posterior_config": posterior_config,
                        "posterior_q0": posterior_samples["q"][0],
                        "posterior_q1": posterior_samples["q"][1],
                        "posterior_w0": posterior_samples["w_mean_over_cells"][0],
                        "posterior_w1": posterior_samples["w_mean_over_cells"][1],
                        "posterior_alpha0": posterior_samples["alpha_mean_over_cells"][0],
                        "posterior_alpha1": posterior_samples["alpha_mean_over_cells"][1],
                        }
        results_dict.update({'model_'+k: v for k, v in vars(args).items()})
        results_dict.update({'sim_'+k: v for k, v in mdata[args.gex_key].uns['sim_params'].items()})
        results_dict.update(calculate_metrics(y_true, p_pred, assignment))
        results.append(results_dict)

    results = pd.DataFrame(results)
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    results.to_csv(csv_dir + f"/{config}.csv")


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)
    input_dir = os.path.join('benchmarks', args.scenario, 'simulation')
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.h5mu')]

    if args.use_mp:
        nworkers = get_slurm_cpu_count()
        mem_per_worker_mb = guess_worker_mem_limit_mb(nworkers)

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(nworkers, initializer=init_worker, initargs=(mem_per_worker_mb,), maxtasksperchild=1,) as pool:
            try:
                list(tqdm(pool.imap_unordered(run_inference_star, [(f, args) for f in input_files]),
                          total=len(input_files), desc="Running inference"))
            except Exception as e:
                # ensures .finished is NOT written and SLURM sees non-zero
                print(f"ERROR: worker failed (likely OOM): {e}", file=sys.stderr)
                sys.exit(1)
    else:
        for f in tqdm(input_files, desc="Running inference"):
            run_inference_star((f, args))

    print("DONE!")


if __name__ == "__main__":
    main()
