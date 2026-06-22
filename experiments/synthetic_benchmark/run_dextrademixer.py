from time import time
t = time()

import argparse
import os
import warnings
import resource

import pandas as pd
import numpyro
import muon as mu

import sys
sys.path.append("../../")

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
sys.path.insert(0, root)

from dextrademixer.model import DextraDemixer
from dextrademixer.utils import convert_str_to_bool_and_none, calculate_metrics, float_or_none, get_cpu_model


posterior_params_full = [
            # Thresholding
            (None, 0.50, None, None,),  # cell-level
            (None, 0.50, True, None,),  # clone-level
            # FDR control
            (0.01, None, None, None,),
            (0.05, None, None, None,),
            (0.10, None, None, None,),
            (0.01, None, True, None,),
            (0.05, None, True, None,),
            (0.10, None, True, None,),
            # credible interval bounded FDR control
            (0.01, None, None, 0.5,),
            (0.05, None, None, 0.5,),
            (0.10, None, None, 0.5,),
            (0.01, None, True, 0.5,),
            (0.05, None, True, 0.5,),
            (0.10, None, True, 0.5,),
        ]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run tool on a single file.")
    # IO parameters
    parser.add_argument("--f_in", type=str, default="benchmarks/scenario_test/simulation/2000_800_0.0_0.9_False_500_None_1.h5mu",
                        help="Path to input h5mu file")
    parser.add_argument("--f_out", type=str, default="benchmarks/scenario_test/csv/2000_800_0.0_0.9_False_500_None_1.h5mu",
                        help="Path to output directory")

    # mudata keys
    parser.add_argument("--gex_key", type=str, default="gex",
                        help="Key for modality where pMHC counts are stored")
    parser.add_argument("--airr_key", type=str, default='airr',
                        help="Key for modality where TCR data is stored")
    parser.add_argument("--pmhc_key", type=str, default="pmhc1",
                        help="Key for pMHC counts, expected to be in 'gex' modality")
    parser.add_argument("--label_key", type=str, default="is_binder",
                        help="Key for labels, expected to be in 'airr' modality")

    # Model parameters
    parser.add_argument("--neg_ctrl_key", type=str, default=None,
                        help="Key for negative control counts. If not None, enables model to use negative control data for "
                             "fitting. Expected to be in 'gex' modality. If None, no negative control data is used.")
    parser.add_argument("--hc_scale_prior", type=float, default=1.0,
                        help="Prior for scale parameter of kmeans model and HalfCauchy for overdispersion model")
    parser.add_argument("--alpha_offset", type=float, default=5.0,
                        help="Additive offset for inverse dispersion parameter of signal component")

    # Optimization parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--maxiter", type=int, default=1000, help="Number of iterations for optimization")
    parser.add_argument("--lr_init_value", type=float, default=3e-1, help="Learning rate")
    parser.add_argument("--lr_end_value", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--lr_decay_rate", type=float, default=0.995, help="Learning rate")
    parser.add_argument("--lr_transition_steps", type=int, default=1, help="Learning rate")
    parser.add_argument("--guide", type=str, default="normal", help="Type of guide for SVI", choices=["normal", "mvnormal"])

    # Logging parameters
    parser.add_argument("--scaling_test", type=str, default=False,
                        help="Use short list of posterior parameters")
    
    # Util
    parser.add_argument("-f", type=str, default=None,
                        help="Unused, for compatibility with Jupyter")

    return parser.parse_args()


def run_inference(args):
    config = os.path.basename(args.f_out.replace('.csv', ''))
    model_config = config.split('-')[0]
    sim_config = config.split('-')[1]
    base_dir = os.path.dirname(args.f_out.replace('csv/', ''))
    if os.path.exists(args.f_out):
        print(f"Results for {args.f_out} already exist, skipping...")
        return

    save_model_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_model_dir, exist_ok=True)
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Running inference for config: {config}")
    numpyro.set_host_device_count(1)

    # Load data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(args.f_in)
    
    # Initialize and fit model
    model = DextraDemixer(model_type='mixturemodelkmeans', mode='I', alpha_model='overdispersion', 
                          model_config={"overdispersion_scale_prior": args.hc_scale_prior, "alpha_offset": args.alpha_offset})
    model.preprocess_model_data(mdata, pmhc_key=args.pmhc_key, gex_key=args.gex_key, neg_ctrl_key=args.neg_ctrl_key,
                                ir_clone_key=None, )
    opt_params = {"maxiter": args.maxiter, "nof_inits": 10,
                  "adam": {"init_value": args.lr_init_value, "end_value": args.lr_end_value,
                           "decay_rate": args.lr_decay_rate, "transition_steps": args.lr_transition_steps}, }
    model.fit_svi(guide=args.guide, svi_config=opt_params, nof_inits=opt_params["nof_inits"], rng_key=args.seed)
    model.save_model(os.path.join(save_model_dir, f"{config}.pkl"))
    
    # Log metrics using different posterior prediction methods
    y_true = mdata.mod[args.airr_key].obs[args.label_key].astype(int).values

    results = []
    # Predict posterior class with different methods, (target_fdr, threshold, clone_median_p, cred_intvl)
    if args.scaling_test:
        posterior_params = [(None, 0.50, True, None,)]  # clone-level thresholding only to test scaling
    else:
        posterior_params = posterior_params_full

    for target_fdr, threshold, clone_median_p, cred_intvl in posterior_params:
        p_pred, assignment = model.predict_posterior_class(target_fdr=target_fdr, threshold=threshold,
                                                           cred_intvl=cred_intvl,
                                                           clonotype_median_p=clone_median_p,
                                                           clone_id=mdata[args.airr_key].obs['clone_id'].values
                                                           )

        results_dict = {'config': config, 'model_config': f'{model_config}{"+clone" if clone_median_p else ""}', 'sim_config': sim_config,
                        'target_fdr': target_fdr, 'threshold': threshold, 'cred_intvl': cred_intvl, 'clone_median_p': clone_median_p, }
        results_dict.update(calculate_metrics(y_true, p_pred, assignment, full_metrics=False))
        if args.scaling_test:
            results_dict.update({"total_time": time() - t, "cpu_model": get_cpu_model(), 
                                 "peak_mem_resource": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,})
        results_dict.update({'model_'+k: v for k, v in vars(args).items()})
        results_dict.update({'sim_'+k: mdata[args.gex_key].uns['sim_params'][k] for k in ['binding_ratio', 'mean_inc', 'nof_clones', 'p_binding_outlier', 'rep', 'rng_key', 'total_cells']})
        results.append(results_dict)
        pd.DataFrame(results).to_csv(args.f_out.replace('.csv', '_intermediate.csv'))

    results = pd.DataFrame(results)
    results.to_csv(args.f_out)
    os.remove(args.f_out.replace('.csv', '_intermediate.csv'))


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    run_inference(args)
    print("DONE!")


if __name__ == "__main__":
    main()
