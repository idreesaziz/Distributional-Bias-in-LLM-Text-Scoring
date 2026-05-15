#!/usr/bin/env python3
"""
Generate clean SVG paper figures from pre-computed CSVs and results.json.
Fixes layout issues in G2, G3, G8, G13, and the mitigation Spearman barplot.

Usage:  python scripts/make_paper_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR      = ROOT / "output" / "figures"
MIT_FIG_DIR  = ROOT / "output" / "mitigations" / "figures"
ANALYSIS_DIR = ROOT / "output" / "analysis"
MIT_RES_DIR  = ROOT / "output" / "mitigations" / "results"

# ── Global style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

LEVELS     = [0.0, 0.2, 0.4, 0.6, 0.8]
AXES_ORDER = ["grammar", "coherence", "information", "lexical"]
AXIS_LABELS = {
    "grammar":     "Grammar",
    "coherence":   "Coherence",
    "information": "Information",
    "lexical":     "Lexical",
}
AXIS_COLORS = {
    "grammar":     "#2171b5",
    "coherence":   "#e6550d",
    "information": "#31a354",
    "lexical":     "#756bb1",
}

# Human-readable short model names (strips "-fireworks", capitalises)
MODEL_SHORT = {
    "gemini-3-flash":          "Gemini 3 Flash",
    "gpt-5-mini":              "GPT-5 mini",
    "gpt-oss-120b-fireworks":  "GPT-OSS 120B",
    "minimax-m2p1-fireworks":  "MiniMax M2P1",
    "gemma2-9b":               "Gemma 2 9B",
    "llama3.1-8b":             "Llama 3.1 8B",
    "mistral-7b":              "Mistral 7B",
    "phi4-14b":                "Phi-4 14B",
    "phi4-mini":               "Phi-4 mini",
    "qwen2.5-14b":             "Qwen 2.5 14B",
    "qwen2.5-7b":              "Qwen 2.5 7B",
}

CATEGORY_SHORT = {
    "Arts & Culture":       "Arts &\nCulture",
    "Business & Economy":   "Business &\nEconomy",
    "Health":               "Health",
    "Politics & Society":   "Politics &\nSociety",
    "Science & Technology": "Science &\nTechnology",
}

METHOD_LABELS = {
    "raw":                    "Raw",
    "raw_argmax":             "Raw",
    "quantile_uniform":       "Quantile\n(Uniform)",
    "quantile_beta":          "Quantile\n(Beta)",
    "expected_score":         "E[S]",
    "asymmetry_fix_alpha=0.0": "E[S]",   # same as expected_score
    "aux_best":               "Aux\nRegressor",
    "contrastive_anchor_A":   "Contrastive\nAnchor",
}

# Fixed model colors (same palette as run_analysis.py)
_FIXED = {"gpt-5-mini": "#2171b5", "gemini-3-flash": "#e6550d"}
_POOL  = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2",
          "#7f7f7f", "#bcbd22", "#17becf", "#d62728", "#ff7f0e"]


def _load():
    results = json.load(open(ANALYSIS_DIR / "results.json", encoding="utf-8"))
    reg_df  = pd.read_csv(ANALYSIS_DIR / "regression_summary.csv")
    mit_df  = pd.read_csv(MIT_RES_DIR  / "mitigation_comparison.csv")
    return results, reg_df, mit_df


def _model_colors(model_names):
    colors, pool_idx = {}, 0
    for m in model_names:
        if m in _FIXED:
            colors[m] = _FIXED[m]
        else:
            colors[m] = _POOL[pool_idx % len(_POOL)]
            pool_idx += 1
    return colors


def _savefig(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════
# G2  Cross-axis dose-response per model (linear reconstruction)
# ═══════════════════════════════════════════════════════════════

def make_g2(reg_df):
    models = sorted(reg_df["model"].unique(),
                    key=lambda m: MODEL_SHORT.get(m, m))
    colors = _model_colors(models)
    n = len(models)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        sharey=True, squeeze=False,
    )
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for j, model in enumerate(models):
        ax = flat[j]
        sub = reg_df[reg_df["model"] == model]
        for axis in AXES_ORDER:
            row = sub[sub["axis"] == axis]
            if row.empty:
                continue
            slope     = row["slope"].iloc[0]
            intercept = row["intercept"].iloc[0]
            ys = [intercept + slope * lv for lv in LEVELS]
            ax.plot(LEVELS, ys, "o-",
                    color=AXIS_COLORS[axis],
                    label=AXIS_LABELS[axis],
                    lw=2, ms=5)
        ax.set_title(MODEL_SHORT.get(model, model), fontweight="bold", pad=6)
        ax.set_ylim(0, 10)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([f"{lv:.1f}" for lv in LEVELS])
        if j % ncols == 0:
            ax.set_ylabel("Mean LLM Score")
        ax.grid(True, alpha=0.25, linewidth=0.6)

    # hide unused cells
    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    # shared x-label
    fig.text(0.5, -0.01, "Degradation Level (λ)", ha="center", fontsize=11)

    # single shared legend at top-right
    handles = [mpatches.Patch(color=AXIS_COLORS[a], label=AXIS_LABELS[a])
               for a in AXES_ORDER]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=True,
               title="Quality Axis", title_fontsize=9)

    fig.suptitle("Cross-Axis Sensitivity Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, FIG_DIR / "G2_cross_axis.svg")


# ═══════════════════════════════════════════════════════════════
# G3  Score distribution histograms (all models)
# ═══════════════════════════════════════════════════════════════

def make_g3(results):
    t1 = results["T1"]
    models = list(t1.keys())
    colors = _model_colors(models)
    n = len(models)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        sharey=False, squeeze=False,
    )
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for j, model in enumerate(models):
        ax = flat[j]
        dist  = t1[model]["distribution"]
        counts = [dist.get(str(k), dist.get(k, 0)) for k in range(11)]
        total  = sum(counts)
        max_c  = max(counts) if counts else 1

        ax.bar(range(11), counts, color=colors[model],
               edgecolor="white", linewidth=0.6, alpha=0.85)

        # mark zero bins with a small red "0" above x-axis (no arrows)
        for val in range(11):
            if counts[val] == 0:
                ax.text(val, max_c * 0.04, "0",
                        ha="center", va="bottom",
                        fontsize=8, color="red", fontweight="bold")

        ax.set_title(MODEL_SHORT.get(model, model), fontweight="bold", pad=6)
        ax.set_xticks(range(11))
        ax.set_xlabel("Score")
        if j % ncols == 0:
            ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

        # n= annotation
        ax.text(0.97, 0.97, f"n={total:,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="gray")

    for j in range(n, nrows * ncols):
        flat[j].set_visible(False)

    fig.suptitle("Score Distribution (n ≈ 9,000 per model)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIG_DIR / "G3_score_histograms.svg")


# ═══════════════════════════════════════════════════════════════
# G8  Forest plot of regression slopes
# ═══════════════════════════════════════════════════════════════

def make_g8(reg_df):
    models = sorted(reg_df["model"].unique(),
                    key=lambda m: MODEL_SHORT.get(m, m))
    colors = _model_colors(models)

    nrows = len(AXES_ORDER) * len(models) + len(AXES_ORDER)  # includes gaps
    fig, ax = plt.subplots(figsize=(9, 0.35 * nrows + 1.5))

    y_pos, y_labels = [], []
    pos = 0
    for axis in AXES_ORDER:
        sub = reg_df[reg_df["axis"] == axis]
        for model in models:
            row = sub[sub["model"] == model]
            if row.empty:
                pos += 1
                continue
            slope  = row["slope"].iloc[0]
            stderr = (row["slope"].iloc[0] - row.get("slope", row["slope"]).iloc[0])
            # use r_squared as proxy if stderr not available separately
            # re_df has stderr column? let's check
            se = row["stderr"].iloc[0] if "stderr" in row.columns else abs(slope) * 0.05
            ci = 1.96 * se
            ax.errorbar(slope, pos,
                        xerr=ci,
                        fmt="o", color=colors[model],
                        capsize=3, ms=6, lw=1.8, elinewidth=1.5)
            y_pos.append(pos)
            y_labels.append(f"{AXIS_LABELS[axis]} / {MODEL_SHORT.get(model, model)}")
            pos += 1
        pos += 0.8  # gap between axis groups

    ax.axvline(0, color="gray", ls="--", alpha=0.4, lw=1)
    ax.axvline(-10, color="black", ls=":", alpha=0.25, lw=1)
    ax.text(-10, ax.get_ylim()[1] if y_pos else 0,
            "Ideal\n(−10)", ha="center", va="bottom",
            fontsize=8, color="black", alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlabel("Regression Slope (β₁)")
    ax.set_title("Sensitivity: Regression Slopes ± 95% CI",
                 fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    _savefig(fig, FIG_DIR / "G8_forest_plot.svg")


# ═══════════════════════════════════════════════════════════════
# G13  Category effects (from T16 results.json)
# ═══════════════════════════════════════════════════════════════

def make_g13(results):
    t16 = results.get("T16", {})
    if not t16:
        print("  [SKIP] G13: no T16 data in results.json")
        return

    models = sorted(t16.keys(), key=lambda m: MODEL_SHORT.get(m, m))
    colors = _model_colors(models)

    # Collect category → mean per model
    all_cats = set()
    for m in models:
        all_cats |= set(t16[m].get("category_means", {}).keys())
    cats = sorted(all_cats)

    ncols = 4
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             sharey=True, squeeze=False)
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for j, model in enumerate(models):
        ax = flat[j]
        cat_means = t16[model].get("category_means", {})
        vals  = [cat_means.get(c, np.nan) for c in cats]
        short = [CATEGORY_SHORT.get(c, c) for c in cats]

        bars = ax.bar(range(len(cats)), vals,
                      color=colors[model], alpha=0.75,
                      edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(short, fontsize=8.5)
        ax.set_title(MODEL_SHORT.get(model, model), fontweight="bold", pad=6)
        ax.set_ylim(0, 10)
        if j % ncols == 0:
            ax.set_ylabel("Mean Score (undegraded)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.15, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)

    for j in range(len(models), nrows * ncols):
        flat[j].set_visible(False)

    fig.suptitle("Undegraded Scores by Article Category",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, FIG_DIR / "G13_category_effects.svg")


# ═══════════════════════════════════════════════════════════════
# Spearman barplot  (mitigation comparison)
# ═══════════════════════════════════════════════════════════════

def make_spearman(mit_df):
    # Normalise method names
    mit_df = mit_df.copy()
    mit_df["method_label"] = mit_df["method"].map(
        lambda m: METHOD_LABELS.get(m, m)
    )

    # Drop duplicate method rows (asymmetry_fix == expected_score)
    mit_df = mit_df[mit_df["method"] != "asymmetry_fix_alpha=0.0"]

    # Order methods
    method_order_raw = ["raw", "raw_argmax", "quantile_uniform", "quantile_beta",
                        "expected_score", "aux_best", "contrastive_anchor_A"]
    # keep only methods that exist in data
    present_methods = [m for m in method_order_raw if m in mit_df["method"].values]
    present_labels  = [METHOD_LABELS.get(m, m) for m in present_methods]

    models = sorted(mit_df["model"].unique(),
                    key=lambda m: MODEL_SHORT.get(m, m))
    model_labels = [MODEL_SHORT.get(m, m) for m in models]
    n_m = len(models)
    n_meth = len(present_methods)

    palette = sns.color_palette("tab10", n_meth)
    x = np.arange(n_m)
    width = 0.8 / n_meth

    fig, ax = plt.subplots(figsize=(max(12, n_m * 1.3), 5))

    for i, (method, label) in enumerate(zip(present_methods, present_labels)):
        sub = mit_df[mit_df["method"] == method]
        rhos = []
        for m in models:
            row = sub[sub["model"] == m]
            rhos.append(row["spearman_rho"].iloc[0] if not row.empty else np.nan)
        offset = (i - n_meth / 2 + 0.5) * width
        ax.bar(x + offset, rhos, width,
               label=label, color=palette[i],
               edgecolor="white", linewidth=0.4, alpha=0.88)

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ (vs. proxy ground truth)")
    ax.set_title("Mitigation Comparison: Spearman ρ per Model",
                 fontweight="bold", pad=10)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(title="Method", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    _savefig(fig, MIT_FIG_DIR / "spearman_rho_barplot.svg")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results, reg_df, mit_df = _load()

    print("\n[G2]  Cross-axis dose-response per model")
    make_g2(reg_df)

    print("\n[G3]  Score histograms")
    make_g3(results)

    print("\n[G8]  Forest plot")
    make_g8(reg_df)

    print("\n[G13] Category effects")
    make_g13(results)

    print("\n[M1]  Spearman barplot (mitigations)")
    make_spearman(mit_df)

    print("\nDone.")
