#!/usr/bin/env python3
"""Model Size vs Score Compression Analysis"""

import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "output" / "figures"
ANALYSIS_DIR = ROOT / "output" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from analysis import build_sample_metadata

MODEL_SIZE_B = {
    "phi4-mini": 3.8, "mistral-7b": 7.0, "qwen2.5-7b": 7.0, "llama3.1-8b": 8.0,
    "gemma2-9b": 9.0, "phi4-14b": 14.0, "qwen2.5-14b": 14.0,
    "gpt-oss-120b-fireworks": 116.8, "minimax-m2p1-fireworks": 228.7,
    "gpt-5-mini": 100.0, "gemini-3-flash": 100.0,
}
API_MODELS = {"gpt-5-mini", "gemini-3-flash"}
AXES_ORDER = ["grammar", "coherence", "information", "lexical"]
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
IDEAL_RANGE = 8.0


def load_scores():
    meta = build_sample_metadata(ROOT)
    scores_dir = ROOT / "data" / "scores"
    rows = []
    for sf in sorted(scores_dir.glob("*.json")):
        records = json.load(open(sf, encoding="utf-8"))
        if not records: continue
        model_name = records[0]["model"]
        for r in records:
            sid = r["sample_id"]
            if sid not in meta or r.get("score") is None: continue
            rows.append({**meta[sid], "sample_id": sid, "model": model_name, "score": r["score"]})
    return pd.DataFrame(rows)


def compute_metrics(df):
    results = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        size = MODEL_SIZE_B.get(model)
        if size is None: continue
        is_api = model in API_MODELS
        mean_0 = sub[sub["level"] == 0.0]["score"].mean()
        mean_08 = sub[sub["level"] == 0.8]["score"].mean()
        compression = (mean_0 - mean_08) / IDEAL_RANGE
        slopes = {}
        for axis in AXES_ORDER:
            ax_sub = sub[sub["axis"] == axis]
            if len(ax_sub) < 5: continue
            r = sp_stats.linregress(ax_sub["level"], ax_sub["score"])
            slopes[axis] = r.slope
        avg_slope = float(np.mean(list(slopes.values()))) if slopes else float("nan")
        axis_compression = {}
        for axis in AXES_ORDER:
            ax_sub = sub[sub["axis"] == axis]
            m0 = ax_sub[ax_sub["level"] == 0.0]["score"].mean()
            m8 = ax_sub[ax_sub["level"] == 0.8]["score"].mean()
            axis_compression[axis] = (m0 - m8) / IDEAL_RANGE
        results.append({
            "model": model, "size_b": size, "is_api": is_api,
            "compression_ratio": float(compression), "avg_slope": avg_slope,
            "score_std": float(sub["score"].std()), "range_used": len(sub["score"].unique()),
            **{f"compression_{a}": axis_compression[a] for a in AXES_ORDER},
        })
    return sorted(results, key=lambda x: x["size_b"])


def fig_scatter(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    local = [m for m in metrics if not m["is_api"]]
    api = [m for m in metrics if m["is_api"]]
    for ax, ykey, ylabel, title, ref_val, ref_label in [
        (axes[0], "compression_ratio", "Compression Ratio", "Score Compression vs Model Size", 1.0, "Perfect (1.0)"),
        (axes[1], "avg_slope", "Avg Sensitivity Slope", "Sensitivity Slope vs Model Size", -10.0, "Perfect (-10)"),
    ]:
        xs_local = [m["size_b"] for m in local]; ys_local = [m[ykey] for m in local]
        ax.scatter(xs_local, ys_local, s=120, c="#1f77b4", edgecolors="white", zorder=3, label="Local/open-weight")
        for m in local:
            ax.annotate(m["model"], (m["size_b"], m[ykey]), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#1f77b4")
        if api:
            xs_api = [m["size_b"] for m in api]; ys_api = [m[ykey] for m in api]
            ax.scatter(xs_api, ys_api, s=120, c="#e6550d", marker="D", edgecolors="white", zorder=3, label="API (est. size)")
            for m in api:
                ax.annotate(m["model"], (m["size_b"], m[ykey]), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#e6550d")
        ax.axhline(ref_val, ls="-.", color="green", alpha=0.4, lw=1, label=ref_label)
        ax.set_xscale("log"); ax.set_xlabel("Model Size (B, log scale)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Model Scale vs Score Compression", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIG_DIR / "size_compression_scatter.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path.relative_to(ROOT)}")


def fig_multiaxis(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    axis_colors = {"grammar": "#1f77b4", "coherence": "#ff7f0e", "information": "#2ca02c", "lexical": "#d62728"}
    markers = {"grammar": "o", "coherence": "s", "information": "^", "lexical": "D"}
    for axis in AXES_ORDER:
        xs = [m["size_b"] for m in metrics]; ys = [m[f"compression_{axis}"] for m in metrics]
        ax.scatter(xs, ys, s=90, color=axis_colors[axis], marker=markers[axis],
                   label=axis.title(), edgecolors="white", zorder=3)
    ax.axhline(1.0, ls="-.", color="green", alpha=0.4, lw=1, label="Perfect (1.0)")
    ax.set_xscale("log"); ax.set_xlabel("Model Size (B, log scale)"); ax.set_ylabel("Compression Ratio")
    ax.set_title("Per-Axis Compression vs Model Size", fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "size_compression_multiaxis.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path.relative_to(ROOT)}")


def fig_table(metrics):
    cols = ["Model", "Size (B)", "Type", "Comp.\nRatio", "Avg\nSlope", "Gram.", "Coh.", "Info.", "Lex.", "s(score)"]
    rows = [[m["model"], f"{m['size_b']:.1f}", "API" if m["is_api"] else "Open",
             f"{m['compression_ratio']:.3f}", f"{m['avg_slope']:.2f}",
             f"{m['compression_grammar']:.3f}", f"{m['compression_coherence']:.3f}",
             f"{m['compression_information']:.3f}", f"{m['compression_lexical']:.3f}",
             f"{m['score_std']:.2f}"] for m in metrics]
    fig, ax = plt.subplots(figsize=(16, 0.5 + 0.45 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2171b5"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        color = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(len(cols)): tbl[i+1, j].set_facecolor(color)
    fig.suptitle("Model Size vs Score Compression Summary", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = FIG_DIR / "size_compression_table.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {path.relative_to(ROOT)}")


def main():
    print("Loading data...")
    df = load_scores()
    print("Computing metrics...")
    metrics = compute_metrics(df)
    print("Generating figures...")
    fig_scatter(metrics)
    fig_multiaxis(metrics)
    fig_table(metrics)
    print("Done.")


if __name__ == "__main__":
    main()