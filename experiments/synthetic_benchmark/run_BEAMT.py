import sys
sys.path.append("../../")

import pandas as pd

from dextramixer.model import BEAMT
import muon as mu


def main(f_in, f_out_csv):
    mdata = mu.read(f_in)
    true_binder = mdata.mod["airr"].obs["is_binder"]
    N = len(true_binder)

    d = {"model": [], "thresh": [], "p": [], "assignment": [], "true_binder": []}

    mixer = BEAMT()
    mixer.preprocess_model_data(mdata, "pmhc1", neg_ctrl_key="neg_control")
    mixer.fit()
    p_thr, assignment_thr = mixer.predict_posterior_class(threshold=0.5)
    p_fdr, assignment_fdr = mixer.predict_posterior_class(target_fdr=0.05)

    d["model"].extend([f"BEAMT"] * N)
    d["thresh"].extend(["thresh"] * N)
    d["p"].extend(p_thr)
    d["assignment"].extend(assignment_thr)
    d["true_binder"].extend(true_binder)

    d["model"].extend(["BEAMT"] * N)
    d["thresh"].extend(["fdr"] * N)
    d["p"].extend(p_fdr)
    d["assignment"].extend(assignment_fdr)
    d["true_binder"].extend(true_binder)

    df = pd.DataFrame.from_dict(d)
    df.to_csv(f_out_csv)


if __name__ == "__main__":
    f_in = sys.argv[1]
    f_out_csv = sys.argv[2]
    main(f_in, f_out_csv)