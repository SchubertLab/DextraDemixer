import argparse
import warnings
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from dextrademixer.model import ITRAP
from dextrademixer.utils import convert_str_to_bool_and_none
import muon as mu


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
    parser.add_argument("--neg_ctrl_key", type=str, default='neg_control', help="Negative control parameter")
    parser.add_argument("--ir_clone_key", type=str, default='clone_id', help="Clonotype parameter")

    # Optimization parameters
    parser.add_argument("--threads", type=int, default=None, help="Number of threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def run_inference(f_in, args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(f_in)

    mixer = ITRAP(umi_cols=None, umi_count_TRA=None, umi_count_TRB=None, filters=None)
    mixer.preprocess_model_data(mdata,
                                pmhc_key=args.pmhc_key,
                                gex_key=args.gex_key,
                                neg_ctrl_key=args.neg_ctrl_key,
                                ir_clone_key=args.ir_clone_key)

    mixer.fit()
    p_pred, assignment = mixer.predict_posterior_class()
    y_true = mdata.mod["airr"].obs[args.label_key]

    print("Classification Report:")
    print(classification_report(y_true[y_true.notna()].astype(int), assignment[y_true.notna()]))

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 2, 1)
    sns.histplot(x=mixer.data["umi_count_mhc"], hue=assignment,
                 discrete=True, element="step", alpha=0.7)
    sns.despine()
    plt.title("Predicted class assignment")

    plt.subplot(2, 2, 2)
    sns.histplot(x=mixer.data["umi_count_mhc"], hue=y_true,
                 discrete=True, element="step", alpha=0.7)
    sns.despine()
    plt.title("True class assignment")

    plt.subplot(2, 2, 3)
    sns.histplot(x=mixer.data["umi_count_mhc"], hue=assignment,
                 discrete=True, element="step", alpha=0.7)
    sns.despine()
    plt.yscale("log")
    plt.title("Predicted class assignment log-scale")

    plt.subplot(2, 2, 4)
    sns.histplot(x=mixer.data["umi_count_mhc"], hue=y_true,
                 discrete=True, element="step", alpha=0.7)
    sns.despine()
    plt.yscale("log")
    plt.title("True class assignment log-scale")

    plt.show()

    return y_true, p_pred, assignment


def main():
    args = parse_arguments()
    args = convert_str_to_bool_and_none(args)

    y_true, p_pred, assignment = run_inference(args.input_file, args)

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
