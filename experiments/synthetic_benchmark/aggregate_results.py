import glob
import sys
import os
import re

import numpy as np
import pandas as pd
import tqdm

from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, confusion_matrix

import warnings
warnings.filterwarnings("error")

def parse_filename(filename):
    # Remove the extension from the filename
    base_name = filename.rsplit('.', 1)[0]

    # Define the regex pattern
    pattern = r"(?P<method>^[a-zA-Z]+)_(?P<key_values>(?:[a-zA-Z]+[\d\.]+_*)+)"

    # Match the pattern in the base name
    match = re.match(pattern, base_name)

    if not match:
        return None

    method = match.group("method")
    key_values_str = match.group("key_values")

    # Define another regex pattern to extract key-value pairs
    key_value_pattern = r"([a-zA-Z]+)([\d\.]+)"
    key_value_matches = re.findall(key_value_pattern, key_values_str)

    # Convert key-value matches to a dictionary
    key_values = {key: float(value) if '.' in value else int(value) for key, value in key_value_matches}

    return method, key_values


def main(f_in_dir, f_out):
    d = {"model":[], "threshold":[], "rep":[], "ncell":[], "nclone":[], "pbinder":[], "cov":[], "meaninc":[], "varinc":[],
         "wprecision":[], "wrecall":[], "wf1":[], "acc":[], "wauc":[], "mcc":[], "tpr":[], "fdr":[], }

    errors = {"file":[], "model":[]}
    files = glob.glob(os.path.join(f_in_dir, "*"))
    for f in tqdm.tqdm(files):
        #print("file:", f, os.path.basename(f))
        #print()
        model_name, sim_params = parse_filename(os.path.basename(f))

        df = pd.read_csv(f)
        for name, group in df.groupby(["model", "thresh"]):

            y_true = group["true_binder"]
            y_pred = group["assignment"]
            p_pred = group["p"]

            if np.isnan(p_pred).any():
                errors["file"].append(f)
                errors["model"].append(name[0])
                continue

            cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            auc = roc_auc_score(y_true, p_pred, average="weighted")
            mcc = matthews_corrcoef(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn)
            fdr = 1 - cr["weighted avg"]["precision"]

            d["model"].append(name[0])
            d["threshold"].append(name[1])
            d["rep"].append(sim_params.get("rep", None))
            d["ncell"].append(sim_params.get("ncell", None))
            d["nclone"].append(sim_params.get("nclone", None))
            d["pbinder"].append(sim_params.get("pbinder", None))
            d["cov"].append(sim_params.get("cov", None))
            d["meaninc"].append(sim_params.get("meaninc", None))
            d["varinc"].append(sim_params.get("varinc", None))

            d["wprecision"].append(cr["weighted avg"]["precision"])
            d["wrecall"].append(cr["weighted avg"]["recall"])
            d["wf1"].append(cr["weighted avg"]["f1-score"])
            d["acc"].append(cr["accuracy"])
            d["wauc"].append(auc)
            d["mcc"].append(mcc)
            d["tpr"].append(tpr)
            d["fdr"].append(fdr)

    df = pd.DataFrame.from_dict(d)
    df.to_csv(f_out)

    df_error = pd.DataFrame.from_dict(errors)
    df_error.to_csv(f_out+".errors")

if __name__ == "__main__":
    f_in_dir = sys.argv[1]
    f_out = sys.argv[2]
    main(f_in_dir, f_out)