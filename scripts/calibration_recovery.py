"""Calibration Recovery Analysis"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "data" / "degraded" / "degraded_samples.json"
GPT_SCORES_PATH = ROOT / "data" / "scores" / "gpt5_mini_scores.json"
GEMINI_SCORES_PATH = ROOT / "data" / "scores" / "llm_scores_gemini.json"
FIG_PATH = ROOT / "output" / "figures" / "G15_calibration_recovery.svg"
RESULTS_PATH = ROOT / "output" / "analysis" / "calibration_results.json"

sys.path.insert(0, str(ROOT / "src"))
from analysis import build_sample_metadata

samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
gpt_scores = json.loads(GPT_SCORES_PATH.read_text(encoding="utf-8"))
gemini_scores = json.loads(GEMINI_SCORES_PATH.read_text(encoding="utf-8"))

n = len(samples)
meta = build_sample_metadata(ROOT)
article_ids = np.array([meta[i]["article_id"] for i in range(n)])
levels = np.array([s["level"] for s in samples])
gpt = np.array([gpt_scores[i]["score"] for i in range(n)])
gem = np.array([gemini_scores[i]["score"] for i in range(n)])
ideal = 10.0 * (1.0 - levels)

unique_article_ids = sorted(set(article_ids.tolist()))
rng = np.random.RandomState(42)
rng.shuffle(unique_article_ids)
split = int(0.8 * len(unique_article_ids))
train_articles = set(unique_article_ids[:split])
test_articles = set(unique_article_ids[split:])
train_mask = np.array([a in train_articles for a in article_ids])
test_mask = ~train_mask


def affine(x, a, b): return a * x + b
def sigmoid(x, a, b, c, d):
    z = np.clip(-a * (x - b), -500, 500)
    return d + c / (1.0 + np.exp(z))
def _eval_metrics(predicted, ideal_arr, levels_arr):
    tau, _ = kendalltau(ideal_arr, predicted)
    rho, _ = spearmanr(ideal_arr, predicted)
    rmse = np.sqrt(np.mean((predicted - ideal_arr) ** 2))
    mean_by_level = {}
    for lv in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask_lv = levels_arr == lv
        if mask_lv.any(): mean_by_level[lv] = predicted[mask_lv].mean()
    cr = (mean_by_level[0.0] - mean_by_level[0.8]) / 8.0
    return {"kendall_tau": round(tau, 4), "spearman_rho": round(rho, 4),
            "rmse": round(rmse, 4), "compression_ratio": round(cr, 4)}


results = {}
for model_name, scores in [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]:
    model_results = {}
    s_train = scores[train_mask].astype(float)
    ideal_train = ideal[train_mask]
    s_test = scores[test_mask].astype(float)
    ideal_test = ideal[test_mask]
    levels_test = levels[test_mask]
    model_results["raw"] = _eval_metrics(s_test, ideal_test, levels_test)
    popt_aff, _ = curve_fit(affine, s_train, ideal_train)
    a_aff, b_aff = popt_aff
    s_test_aff = affine(s_test, a_aff, b_aff)
    metrics_aff = _eval_metrics(s_test_aff, ideal_test, levels_test)
    metrics_aff["params"] = {"a": round(a_aff, 4), "b": round(b_aff, 4)}
    model_results["affine"] = metrics_aff
    try:
        popt_sig, _ = curve_fit(sigmoid, s_train, ideal_train, p0=[0.5, 5.0, 12.0, -1.0],
                                bounds=([0.05, 0.0, 1.0, -15.0], [5.0, 10.0, 30.0, 10.0]), maxfev=20000)
        s_test_sig = sigmoid(s_test, *popt_sig)
        metrics_sig = _eval_metrics(s_test_sig, ideal_test, levels_test)
        metrics_sig["params"] = {k: round(v, 4) for k, v in zip("abcd", popt_sig)}
        model_results["sigmoid"] = metrics_sig
    except RuntimeError:
        popt_sig = None
        model_results["sigmoid"] = {"error": "convergence failure"}
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train, ideal_train)
    s_test_iso = iso.predict(s_test)
    model_results["isotonic"] = _eval_metrics(s_test_iso, ideal_test, levels_test)
    results[model_name] = model_results

fig, axes_arr = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
level_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
ideal_curve = 10.0 * (1.0 - level_vals)
for ax, (model_name, scores_arr) in zip(axes_arr, [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]):
    mr = results[model_name]
    s_test_local = scores_arr[test_mask].astype(float)
    levels_test_local = levels[test_mask]
    a_aff = mr["affine"]["params"]["a"]; b_aff = mr["affine"]["params"]["b"]
    s_aff_local = affine(s_test_local, a_aff, b_aff)
    if "params" in mr.get("sigmoid", {}):
        sp = mr["sigmoid"]["params"]
        s_sig_local = sigmoid(s_test_local, sp["a"], sp["b"], sp["c"], sp["d"])
    else:
        s_sig_local = None
    iso_local = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_local.fit(scores_arr[train_mask].astype(float), ideal[train_mask])
    s_iso_local = iso_local.predict(s_test_local)
    raw_means, aff_means, sig_means, iso_means = [], [], [], []
    for lv in level_vals:
        mask_lv = levels_test_local == lv
        raw_means.append(s_test_local[mask_lv].mean())
        aff_means.append(s_aff_local[mask_lv].mean())
        iso_means.append(s_iso_local[mask_lv].mean())
        if s_sig_local is not None: sig_means.append(s_sig_local[mask_lv].mean())
    ax.plot(level_vals, ideal_curve, "k--", linewidth=2.5, label="Ideal")
    ax.plot(level_vals, raw_means, "o-", color="#d62728", linewidth=2, markersize=8,
            label=f"Raw (RMSE={mr['raw']['rmse']:.2f})")
    ax.plot(level_vals, aff_means, "s-", color="#2ca02c", linewidth=2, markersize=8,
            label=f"Affine (RMSE={mr['affine']['rmse']:.2f})")
    if sig_means:
        ax.plot(level_vals, sig_means, "^-", color="#1f77b4", linewidth=2, markersize=8,
                label=f"Sigmoid (RMSE={mr['sigmoid']['rmse']:.2f})")
    ax.plot(level_vals, iso_means, "D-", color="#9467bd", linewidth=2, markersize=7,
            label=f"Isotonic (RMSE={mr['isotonic']['rmse']:.2f})")
    ax.fill_between(level_vals, raw_means, ideal_curve, alpha=0.10, color="#d62728")
    ax.set_xlabel("Degradation level", fontsize=13)
    ax.set_title(model_name, fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 0.82); ax.set_ylim(0, 11)
    ax.set_xticks(level_vals); ax.legend(fontsize=9.5, loc="lower left"); ax.grid(True, alpha=0.3)
axes_arr[0].set_ylabel("Score", fontsize=13)
fig.suptitle("Calibration Recovery", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
svg_path = FIG_PATH.with_suffix(".svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {svg_path}")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")