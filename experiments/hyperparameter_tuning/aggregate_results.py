import glob
import sys
import os
import re

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, confusion_matrix


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
    files = glob.glob(os.path.join(f_in_dir, "*"))
    dfs = [pd.read_csv(f) for f in files]

    df.to_csv(f_out)


if __name__ == "__main__":
    f_in_dir = sys.argv[1]
    f_out = sys.argv[2]
    main(f_in_dir, f_out)