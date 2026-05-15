#!/usr/bin/env python3
"""Quantile-Normalisation Mitigation"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analysis import (
    AXES_ORDER, LEVELS, RANDOM_STATE, load_scores, proxy_ground_truth,
    compute_compression_ratio, bootstrap_ci, safe_quantile_ranks, pairwise_accuracy,
)

RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_STATE)


def quantile_normalise_uniform(scores):
    return safe_quantile_ranks(scores) * 10.0


def quantile_normalise_beta(scores, a=2.0, b=2.0):
    qr = np.clip(safe_quantile_ranks(scores), 1e-8, 1 - 1e-8)
    return sp_stats.beta.ppf(qr, a, b) * 10.0


def eval_mitigation(df_model, score_col):
    scores = df_model[score_col].values
    target = proxy_ground_truth(df_model["level"].values)
    wd = float(sp_stats.wasserstein_distance(scores, target))
    cr = compute_compression_ratio(scores)
    rho, rho_p = sp_stats.spearmanr(scores, target)
    slope = sp_stats.linregress(df_model["level"].values, scores).slope
    pa = pairwise_accuracy(scores, target)
    return {
        "wasserstein": round(wd, 4), "compression_ratio": round(cr, 4),
        "spearman_rho": round(float(rho), 4), "spearman_p": float(rho_p),
        "dose_response_slope": round(float(slope), 4), "pairwise_accuracy": round(pa, 4),
    }


def main():
    print("Loading scores …")
    df = load_scores(ROOT)
    models = sorted(df["model"].unique())
    results_uniform, results_beta = [], []
    for model in models:
        sub = df[df["model"] == model].copy()
        raw = sub["score"].values.astype(float)
        sub["score_uniform"] = quantile_normalise_uniform(raw)
        sub["score_beta"] = quantile_normalise_beta(raw)
        m_raw = eval_mitigation(sub, "score"); m_raw.update({"model": model, "method": "raw"})
        m_uni = eval_mitigation(sub, "score_uniform"); m_uni.update({"model": model, "method": "quantile_uniform"})
        m_beta = eval_mitigation(sub, "score_beta"); m_beta.update({"model": model, "method": "quantile_beta"})
        results_uniform.extend([m_raw, m_uni])
        results_beta.extend([m_raw, m_beta])
        print(f"  {model:30s}  CR raw={m_raw['compression_ratio']:.3f}")

    cols = ["model", "method", "wasserstein", "compression_ratio",
            "spearman_rho", "spearman_p", "dose_response_slope", "pairwise_accuracy"]
    pd.DataFrame(results_uniform)[cols].to_csv(RES_DIR / "quantile_uniform.csv", index=False)
    pd.DataFrame(results_beta)[cols].to_csv(RES_DIR / "quantile_beta.csv", index=False)

    n_models = len(models)
    ncols = min(4, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        raw = sub["score"].values.astype(float)
        sns.kdeplot(raw, ax=ax, label="Raw", color="#d62728", linewidth=1.5)
        sns.kdeplot(quantile_normalise_uniform(raw), ax=ax, label="Uniform", color="#2ca02c", linewidth=1.5)
        sns.kdeplot(quantile_normalise_beta(raw), ax=ax, label="Beta(2,2)", color="#1f77b4", linewidth=1.5)
        ax.set_title(model, fontsize=10)
        ax.set_xlim(-0.5, 10.5)
        if idx == 0: ax.legend(fontsize=8)
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Quantile Normalisation: Score Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quantile_kde.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {FIG_DIR / 'quantile_kde.svg'}")
    print("\nDone.")


if __name__ == "__main__":
    main()