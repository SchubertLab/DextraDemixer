import itertools
import sys
import os

sys.path.append("../../")

import itertools as itr
import numpyro
import pandas as pd

from dextrademixer.model import DextraDemixer
import muon as mu


def main(f_in, f_out_csv):
    numpyro.set_host_device_count(4)
    base_path = os.path.basename(f_out_csv)

    mdata = mu.read(f_in)
    true_binder = mdata.mod["airr"].obs["is_binder"]
    N = len(true_binder)

    d = {"model": [], "thresh": [], "p": [], "assignment": [], "true_binder": []}
    for m, neg_ctrl, ir_clone, ir_cov, svi in itr.product(["I", "H", "C"],
                                                     [None, "neg_control"],
                                                     [None, "clone_id"],
                                                     [None, "clone_cov"],
                                                     [True]):

        if "_cov1_" not in f_in and ir_cov is not None and ir_clone is not None:
            continue

        if ir_cov is not None and ir_clone is None:
            continue

        if m == "C" and ir_clone is None:
            continue

        model_config = f"dextramixerkmeans_svi_{int(svi)}_mode_{m}_negctrl_{int(neg_ctrl is not None)}_clone_{int(ir_clone is not None)}_cov_{int(ir_cov is not None)}"

        print(model_config)

        mixer = DextraDemixer(model_type="mixturemodelkmeans", mode=m)
        mixer.preprocess_model_data(mdata, "pmhc1",
                                    neg_ctrl_key=neg_ctrl,
                                    ir_clone_key=ir_clone,
                                    ir_cov_key=ir_cov)
        if svi:
            trace = mixer.fit_svi()
        else:
            trace = mixer.fit()

        # trace.to_netcdf(filename=os.path.join(base_path, "dextramier_"+model_config+"{}.ncdf"))

        p_thr, assignment_thr = mixer.predict_posterior_class(threshold=0.5)
        p_fdr, assignment_fdr = mixer.predict_posterior_class(target_fdr=0.05)
        d["model"].extend([model_config] * N)
        d["thresh"].extend(["thresh"] * N)
        d["p"].extend(p_thr)
        d["assignment"].extend(assignment_thr)
        d["true_binder"].extend(true_binder)

        d["model"].extend([model_config] * N)
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
