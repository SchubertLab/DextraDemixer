import glob
import sys
import os

import pandas as pd


def main(f_in_dir, f_out):
    files = glob.glob(os.path.join(f_in_dir, "*"))
    dfs = [pd.read_csv(f) for f in files]

    extracted_rows=[]
    for name, df in zip(files, dfs):
        if 'value' not in df.columns:
                raise ValueError("Each DataFrame must have a 'value' column.")

        max_row = df.loc[df['value'].idxmax()]
        param_cols = ["model", "value"]+[col for col in df.columns if col.startswith('param')]
        max_row["model"] = name
        extracted_rows.append(max_row[param_cols])

    pd.DataFrame(extracted_rows).to_csv(f_out, index=False)


if __name__ == "__main__":
    f_in_dir = sys.argv[1]
    f_out = sys.argv[2]
    main(f_in_dir, f_out)