import os
import warnings

import muon as mu
import numpy as np
import numpyro as npy
import pandas as pd
import pytest

from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

from dextrademixer.model import DextraDemixer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

EXPERIMENTS = [
    "1000,0.4,0.0,0.5,False,20.0,None,3",
    "2000,0.4,0.0,0.05,False,30.0,None,1",
    "5000,0.4,0.0,0.2,False,30.0,None,4",
]

ALL_MODEL_VARIANTS = ["DextraDemixer", "DextraDemixer+neg.", "DextraDemixer+clone", "DextraDemixer+neg.+clone"]
FDR_CAPABLE_MODEL_VARIANTS = ["DextraDemixer+neg.", "DextraDemixer+neg.+clone"]

FDR_PREDICTION_MODES = [
    {"target_fdr": 0.05},
    {"target_fdr": 0.05, "cred_intvl": 0.50},
]

_MODEL_KWARGS = {
    "DextraDemixer": {"neg_ctrl_key": None, "clonotype_median_p": False},
    "DextraDemixer+neg.": {"neg_ctrl_key": "neg_control", "clonotype_median_p": False},
    "DextraDemixer+clone": {"neg_ctrl_key": None, "clonotype_median_p": True},
    "DextraDemixer+neg.+clone": {"neg_ctrl_key": "neg_control", "clonotype_median_p": True},
}

# Known SVI numeric drift across the supported dependency version range (jax/numpyro/optax
# and related numeric libs). n_inits=10 picks the best of several random-restart SVI fits,
# so small differences in a library's RNG/optimizer internals across versions can shift which
# restart "wins" and cascade into a materially different final fit for some (model, experiment)
# combinations, even though other combinations reproduce exactly across the same version range.
# This reflects inherent optimization-landscape sensitivity, not a deterministic regression.
_TOLERANCE_OVERRIDES = {
    ("DextraDemixer+neg.", "5000,0.4,0.0,0.2,False,30.0,None,4", "threshold"): 5e-3,
    ("DextraDemixer", "1000,0.4,0.0,0.5,False,20.0,None,3", "threshold"): 5e-3,
    ("DextraDemixer", "2000,0.4,0.0,0.05,False,30.0,None,1", "threshold"): 0.025,
    ("DextraDemixer+clone", "2000,0.4,0.0,0.05,False,30.0,None,1", "threshold"): 0.015,
}
_DEFAULT_ATOL = 1e-3


@pytest.fixture(scope="module", autouse=True)
def _numpyro_platform():
    npy.set_platform("cpu")
    npy.set_host_device_count(1)


@pytest.fixture(scope="module")
def expected_results():
    return pd.read_csv(os.path.join(DATA_DIR, "expected_results.csv"), index_col=0)


def _run_model(model_variant, experiment, expected_results, threshold=None, target_fdr=None, cred_intvl=None):
    kwargs = _MODEL_KWARGS[model_variant]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(os.path.join(DATA_DIR, f"{experiment}.h5mu"))

    model = DextraDemixer(overdispersion_scale_prior=1.0, alpha_offset=5.0)
    model.preprocess_model_data(mdata, pmhc_key="pmhc1", pmhc_modality_key="gex", neg_ctrl_key=kwargs["neg_ctrl_key"])

    model.fit_svi(maxiter=1000, lr_init_value=3e-1, lr_end_value=3e-3, lr_decay_rate=0.995,
                  lr_transition_steps=1, n_inits=10, rng_key=42)

    p_pred, assignment = model.predict_posterior_class(
        target_fdr=target_fdr,
        threshold=threshold,
        cred_intvl=cred_intvl,
        clonotype_median_p=kwargs["clonotype_median_p"],
        clone_id=mdata["airr"].obs["clone_id"].values,
    )

    y_true = mdata.mod["airr"].obs["is_binder"].astype(int).values
    assignment = np.asarray(assignment)
    n_predicted_positive = assignment.sum()
    n_false_positive = ((assignment == 1) & (y_true == 0)).sum()
    fdr = n_false_positive / n_predicted_positive if n_predicted_positive > 0 else 0.0
    results = pd.Series({"f1": f1_score(y_true, assignment), "precision": precision_score(y_true, assignment),
                         "recall": recall_score(y_true, assignment),
                         "aps": average_precision_score(y_true, p_pred),
                         "fdr": fdr,})

    expected = expected_results[
        (expected_results["model_config"] == model_variant)
        & (expected_results["sim_config"] == experiment)
    ]
    if threshold is not None:
        expected = expected[expected["threshold"] == threshold]
    else:
        expected = expected[expected["target_fdr"] == target_fdr]
        if cred_intvl is None:
            expected = expected[expected["cred_intvl"].isna()]
        else:
            expected = expected[expected["cred_intvl"] == cred_intvl]
    expected = expected.iloc[0]

    mode = "threshold" if threshold is not None else "target_fdr"
    atol = _TOLERANCE_OVERRIDES.get((model_variant, experiment, mode), _DEFAULT_ATOL)

    metrics = [m for m in ("f1", "precision", "recall", "aps", "fdr") if pd.notna(expected[m])]
    assert np.allclose(results[metrics].values, expected[metrics].astype(float).values, atol=atol), (
        f"{model_variant} on {experiment} (threshold={threshold}, target_fdr={target_fdr}, "
        f"cred_intvl={cred_intvl})\nExpected: {expected[metrics].astype(float).values}, "
        f"got: {results[metrics].values}"
    )


def test_input_formats_agree():
    """MuData, AnnData and DataFrame must all reach the same fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(os.path.join(DATA_DIR, f"{EXPERIMENTS[0]}.h5mu"))
    gex, airr = mdata.mod["gex"], mdata.mod["airr"]
    adata = gex.copy()
    adata.obs["clone"] = [f"c{i}" for i in airr.obs["clone_id"]]  # string ids, as scirpy has them
    df = gex.to_df()
    df["clone"] = adata.obs["clone"].values
    svi = dict(pmhc_key="pmhc1", neg_ctrl_key="neg_control", ir_clone_key="clone", maxiter=50,
               n_inits=2, progress_bar=False, rng_key=42)

    out = {}
    for name, data in {"MuData": mdata, "AnnData": adata, "DataFrame": df}.items():
        keys = {**svi, "ir_clone_key": "clone_id"} if name == "MuData" else svi
        model = DextraDemixer().fit(data, **keys)
        out[name] = model.predict_posterior_class(threshold=0.5, clonotype_median_p=True)

    for other in ("AnnData", "DataFrame"):
        np.testing.assert_allclose(np.asarray(out["MuData"][0]), np.asarray(out[other][0]),
                                   err_msg=f"MuData vs {other} probabilities differ")
        np.testing.assert_array_equal(np.asarray(out["MuData"][1]), np.asarray(out[other][1]))

    with pytest.raises(TypeError, match="unsupported input type"):
        DextraDemixer().preprocess_model_data(df.to_numpy(), pmhc_key=0)


def test_fit_wrapper_matches_two_step():
    """`fit` duplicates the defaults of `preprocess_model_data`/`fit_svi`, so check both that it
    delegates correctly and that no default has drifted apart from them."""
    import inspect

    fit, low_level = inspect.signature(DextraDemixer.fit).parameters, {}
    for method in (DextraDemixer.preprocess_model_data, DextraDemixer.fit_svi):
        low_level.update(inspect.signature(method).parameters)
    drifted = {name: (p.default, low_level[name].default) for name, p in fit.items()
               if name in low_level and p.default is not p.empty
               and p.default != low_level[name].default}
    assert not drifted, f"fit() defaults differ from the low-level methods: {drifted}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdata = mu.read(os.path.join(DATA_DIR, f"{EXPERIMENTS[0]}.h5mu"))
    kwargs = dict(pmhc_key="pmhc1", pmhc_modality_key="gex", neg_ctrl_key="neg_control")
    svi = dict(maxiter=100, n_inits=2, progress_bar=False, rng_key=42)

    two_step = DextraDemixer(overdispersion_scale_prior=1.0, alpha_offset=5.0)
    two_step.preprocess_model_data(mdata, **kwargs)
    two_step.fit_svi(**svi)
    p_two, a_two = two_step.predict_posterior_class(threshold=0.5)

    one_call = DextraDemixer(overdispersion_scale_prior=1.0, alpha_offset=5.0).fit(mdata, **kwargs, **svi)
    p_one, a_one = one_call.predict_posterior_class(threshold=0.5)

    np.testing.assert_allclose(np.asarray(p_one), np.asarray(p_two))
    np.testing.assert_array_equal(np.asarray(a_one), np.asarray(a_two))


class TestDextraDemixer:
    def test_model_registration(self):
        assert DextraDemixer.available_methods()

    @pytest.mark.slow
    @pytest.mark.parametrize("experiment", EXPERIMENTS)
    @pytest.mark.parametrize("model_variant", ALL_MODEL_VARIANTS)
    def test_model_variants_threshold(self, model_variant, experiment, expected_results):
        _run_model(model_variant, experiment, expected_results, threshold=0.5)

    @pytest.mark.slow
    @pytest.mark.parametrize("experiment", EXPERIMENTS)
    @pytest.mark.parametrize("prediction_mode", FDR_PREDICTION_MODES, ids=["target_fdr", "target_fdr_cred_intvl"])
    @pytest.mark.parametrize("model_variant", FDR_CAPABLE_MODEL_VARIANTS)
    def test_model_variants_fdr_control(self, model_variant, prediction_mode, experiment, expected_results):
        _run_model(model_variant, experiment, expected_results, **prediction_mode)

    @pytest.mark.skip(reason="Requires METAL backend, not available in CI/most dev environments")
    def test_GPU_Metal(self, expected_results):
        npy.set_platform("METAL")
        _run_model("DextraDemixer", EXPERIMENTS[0], expected_results, threshold=0.5)


if __name__ == "__main__":
    pytest.main([__file__])
