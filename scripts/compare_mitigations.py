#!/usr/bin/env python3
"""Cross-Method Mitigation Comparison"""

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
sys.path.insert(0, str(ROOT / "scripts"))

from analysis import (
    AXES_ORDER, LEVELS, RANDOM_STATE, load_scores, proxy_ground_truth,
    compute_compression_ratio, pairwise_accuracy,
)

RES_DIR = ROOT / "output" / "mitigations" / "results"
FIG_DIR = ROOT / "output" / "mitigations" / "figures"
INT_DIR = ROOT / "output" / "mitigations" / "intermediate"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)

METRICS = ["wasserstein", "compression_ratio", "spearman_rho",
           "dose_response_slope", "pairwise_accuracy"]


def eval_scores(scores, levels):
    target = proxy_ground_truth(levels)
    wd = float(sp_stats.wasserstein_distance(scores, target))
    cr = compute_compression_ratio(scores)
    rho, rho_p = sp_stats.spearmanr(scores, target)
    slope = sp_stats.linregress(levels, scores).slope
    pa = pairwise_accuracy(scores, target)
    return {
        "wasserstein": round(wd, 4),
        "compression_ratio": round(cr, 4),
        "spearman_rho": round(float(rho), 4),
        "dose_response_slope": round(float(slope), 4),
        "pairwise_accuracy": round(pa, 4),
    }


def collect_all_results(df_raw):
    all_rows = []
    models = sorted(df_raw["model"].unique())
    for model in models:
        sub = df_raw[df_raw["model"] == model]
        m = eval_scores(sub["score"].values.astype(float), sub["level"].values)
        m.update({"model": model, "method": "raw"})
        all_rows.append(m)
    for fname, skip_method in [
        ("quantile_uniform.csv", "raw"),
        ("quantile_beta.csv", "raw"),
        ("logprob_rescaling.csv", "raw_argmax"),
    ]:
        f = RES_DIR / fname
        if f.exists():
            df = pd.read_csv(f)
            for _, row in df[df["method"] != skip_method].iterrows():
                all_rows.append(row.to_dict())
    ar = RES_DIR / "aux_regressor.csv"
    if ar.exists():
        adf = pd.read_csv(ar)
        adf = adf[adf["method"] != "raw"]
        if not adf.empty and "rmse" in adf.columns:
            best = adf.loc[adf.groupby("model")["rmse"].idxmin()]
            for _, row in best.iterrows():
                d = row.to_dict()
                d["method"] = "aux_best"
                all_rows.append(d)
    cd = RES_DIR / "contrastive_delta.csv"
    if cd.exists():
        cdf = pd.read_csv(cd)
        for model in cdf["model"].unique():
            msub = cdf[cdf["model"] == model].dropna(subset=["score_anchor_A"])
            if len(msub) < 10:
                continue
            m = eval_scores(msub["score_anchor_A"].values, msub["level"].values)
            m.update({"model": model, "method": "contrastive_anchor_A"})
            all_rows.append(m)
    return pd.DataFrame(all_rows)


def fig_metric_barplot(comp, metric, ylabel=None):
    if metric not in comp.columns:
        return
    pivot = comp.pivot_table(index="model", columns="method", values=metric, aggfunc="first")
    pivot = pivot.reindex(sorted(pivot.index))
    fig, ax = plt.subplots(figsize=(max(12, len(pivot) * 1.5), 6))
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="white")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_title(f"Mitigation Comparison: {metric.replace('_', ' ').title()}", fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{metric}_barplot.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / f'{metric}_barplot.svg'}")


def fig_lowess_response(df_raw, comp):
    try:
        from generate_graphs import plot_lowess_response_curve
    except ImportError:
        print("  [SKIP] Cannot import plot_lowess_response_curve")
        return
    models = sorted(df_raw["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    ideal_levels = np.array(LEVELS)
    ideal_scores = proxy_ground_truth(ideal_levels)
    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_raw[df_raw["model"] == model]
        plot_lowess_response_curve(sub["level"].values, sub["score"].values.astype(float),
                                   ax=ax, label="Raw", color="#d62728")
        ax.plot(ideal_levels, ideal_scores, "--k", alpha=0.5, linewidth=1.5, label="Ideal")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Degradation Level")
        ax.set_ylabel("Score" if idx % ncols == 0 else "")
        ax.set_ylim(-0.5, 10.5)
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("LOWESS Dose-Response Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lowess_response_curves.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'lowess_response_curves.svg'}")


def fig_delta_compression_heatmap(comp):
    raw_cr = comp[comp["method"] == "raw"].set_index("model")["compression_ratio"]
    non_raw = comp[comp["method"] != "raw"].copy()
    if non_raw.empty:
        return
    non_raw["delta_cr"] = non_raw.apply(
        lambda r: r["compression_ratio"] - raw_cr.get(r["model"], 0), axis=1)
    pivot = non_raw.pivot_table(index="model", columns="method", values="delta_cr", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                ax=ax, cbar_kws={"label": "Δ Compression Ratio"})
    ax.set_title("Change in Compression Ratio vs Raw", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delta_compression_heatmap.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'delta_compression_heatmap.svg'}")


def fig_method_rank_heatmap(comp):
    non_raw = comp.copy()
    if non_raw.empty or "spearman_rho" not in non_raw.columns:
        return
    pivot = non_raw.pivot_table(index="model", columns="method", values="spearman_rho", aggfunc="first")
    ranks = pivot.rank(axis=1, ascending=False)
    fig, ax = plt.subplots(figsize=(max(10, len(ranks.columns) * 1.5), max(6, len(ranks) * 0.5)))
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="YlGnBu_r", ax=ax,
                cbar_kws={"label": "Rank (1=best)"})
    ax.set_title("Method Rank by Spearman ρ", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_rank_heatmap.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'method_rank_heatmap.svg'}")


def fig_scatter_cr_vs_rho(comp):
    if comp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    methods = sorted(comp["method"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(methods), 1)))
    for i, method in enumerate(methods):
        msub = comp[comp["method"] == method]
        ax.scatter(msub["compression_ratio"], msub["spearman_rho"],
                   label=method, color=colors[i], s=60, alpha=0.8, edgecolors="white")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Compression Ratio vs Spearman ρ (All Methods)", fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_cr_vs_rho.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'scatter_cr_vs_rho.svg'}")


def fig_summary_radar(comp):
    methods = sorted(comp["method"].unique())
    if len(methods) < 2:
        return
    metric_cols = [m for m in METRICS if m in comp.columns]
    if not metric_cols:
        return
    avg = comp.groupby("method")[metric_cols].mean()
    norm = avg.copy()
    for col in metric_cols:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn) if mx > mn else 0.5
    if "wasserstein" in norm.columns:
        norm["wasserstein"] = 1 - norm["wasserstein"]
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(methods), 1)))
    for i, method in enumerate(methods):
        if method not in norm.index:
            continue
        values = norm.loc[method].values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=method, color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_cols], fontsize=9)
    ax.set_title("Method Comparison (Normalised)", fontsize=13, pad=20)
    ax.legend(bbox_to_anchor=(1.3, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "summary_radar.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {FIG_DIR / 'summary_radar.svg'}")


def main():
    print("Loading raw scores …")
    df_raw = load_scores(ROOT)
    if "score_probs" in df_raw.columns:
        df_raw = df_raw.drop(columns=["score_probs"])
    print(f"  {len(df_raw)} rows, {len(df_raw['model'].unique())} models")
    print("Collecting all mitigation results …")
    comp = collect_all_results(df_raw)
    comp.to_csv(RES_DIR / "mitigation_comparison.csv", index=False)
    print("Generating figures …")
    for metric in METRICS:
        fig_metric_barplot(comp, metric)
    fig_lowess_response(df_raw, comp)
    fig_delta_compression_heatmap(comp)
    fig_method_rank_heatmap(comp)
    fig_scatter_cr_vs_rho(comp)
    fig_summary_radar(comp)
    print("Done.")


if __name__ == "__main__":
    main()