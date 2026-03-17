"""
Calibration Recovery Analysis
=============================
Fits affine, sigmoid, and isotonic calibration functions to LLM scores,
evaluates on held-out articles, and generates a figure showing raw vs
calibrated dose-response curves against the ideal calibration line.

Output:
  - output/figures/G15_calibration_recovery.png
  - output/analysis/calibration_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "data" / "degraded" / "degraded_samples.json"
GPT_SCORES_PATH = ROOT / "data" / "scores" / "gpt5_mini_scores.json"
GEMINI_SCORES_PATH = ROOT / "data" / "scores" / "llm_scores_gemini.json"
FIG_PATH = ROOT / "output" / "figures" / "G15_calibration_recovery.png"
RESULTS_PATH = ROOT / "output" / "analysis" / "calibration_results.json"

# ── Load data ──────────────────────────────────────────────────
samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
gpt_scores = json.loads(GPT_SCORES_PATH.read_text(encoding="utf-8"))
gemini_scores = json.loads(GEMINI_SCORES_PATH.read_text(encoding="utf-8"))

# Build arrays: for each sample, get title, axis, level, gpt_score, gemini_score
n = len(samples)
titles = [s["source_title"] for s in samples]
levels = np.array([s["level"] for s in samples])
axes = [s["axis"] for s in samples]
gpt = np.array([gpt_scores[i]["score"] for i in range(n)])
gem = np.array([gemini_scores[i]["score"] for i in range(n)])

# Ground truth: ideal score = 10 * (1 - level)
ideal = 10.0 * (1.0 - levels)

# ── Train/test split by article (80/20) ───────────────────────
unique_titles = sorted(set(titles))
rng = np.random.RandomState(42)
rng.shuffle(unique_titles)
split = int(0.8 * len(unique_titles))
train_titles = set(unique_titles[:split])
test_titles = set(unique_titles[split:])

train_mask = np.array([t in train_titles for t in titles])
test_mask = ~train_mask

print(f"Train articles: {len(train_titles)}, Test articles: {len(test_titles)}")
print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")


# ── Calibration functions ──────────────────────────────────────
def affine(x, a, b):
    return a * x + b


def sigmoid(x, a, b, c, d):
    """Generalized sigmoid: d + c / (1 + exp(-a * (x - b)))"""
    z = np.clip(-a * (x - b), -500, 500)  # prevent overflow
    return d + c / (1.0 + np.exp(z))


def _eval_metrics(predicted, ideal_arr, levels_arr):
    """Compute RMSE, Kendall tau, Spearman rho, and compression ratio."""
    tau, _ = kendalltau(ideal_arr, predicted)
    rho, _ = spearmanr(ideal_arr, predicted)
    rmse = np.sqrt(np.mean((predicted - ideal_arr) ** 2))
    mean_by_level = {}
    for lv in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mask_lv = levels_arr == lv
        if mask_lv.any():
            mean_by_level[lv] = predicted[mask_lv].mean()
    cr = (mean_by_level[0.0] - mean_by_level[0.8]) / 8.0
    return {
        "kendall_tau": round(tau, 4),
        "spearman_rho": round(rho, 4),
        "rmse": round(rmse, 4),
        "compression_ratio": round(cr, 4),
    }


# ── Fit and evaluate ──────────────────────────────────────────
results = {}

for model_name, scores in [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]:
    model_results = {}

    # Train data
    s_train = scores[train_mask].astype(float)
    ideal_train = ideal[train_mask]

    # Test data
    s_test = scores[test_mask].astype(float)
    ideal_test = ideal[test_mask]
    levels_test = levels[test_mask]

    # --- Raw (uncalibrated) metrics on test set ---
    model_results["raw"] = _eval_metrics(s_test, ideal_test, levels_test)

    # --- Affine calibration ---
    popt_aff, _ = curve_fit(affine, s_train, ideal_train)
    a_aff, b_aff = popt_aff
    s_test_aff = affine(s_test, a_aff, b_aff)

    metrics_aff = _eval_metrics(s_test_aff, ideal_test, levels_test)
    metrics_aff["params"] = {"a": round(a_aff, 4), "b": round(b_aff, 4)}
    model_results["affine"] = metrics_aff

    # --- Sigmoid calibration (with proper bounds) ---
    # p0: a=steepness, b=midpoint, c=range, d=lower asymptote
    # We want the sigmoid to span roughly [0, 12] centered at raw score ~5
    try:
        popt_sig, _ = curve_fit(
            sigmoid, s_train, ideal_train,
            p0=[0.5, 5.0, 12.0, -1.0],
            bounds=([0.05, 0.0, 1.0, -15.0], [5.0, 10.0, 30.0, 10.0]),
            maxfev=20000,
        )
        s_test_sig = sigmoid(s_test, *popt_sig)

        metrics_sig = _eval_metrics(s_test_sig, ideal_test, levels_test)
        metrics_sig["params"] = {k: round(v, 4) for k, v in zip("abcd", popt_sig)}
        model_results["sigmoid"] = metrics_sig
    except RuntimeError:
        print(f"  Sigmoid fit failed for {model_name}")
        popt_sig = None
        model_results["sigmoid"] = {"error": "convergence failure"}

    # --- Isotonic regression (nonparametric monotonic upper bound) ---
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(s_train, ideal_train)
    s_test_iso = iso.predict(s_test)

    metrics_iso = _eval_metrics(s_test_iso, ideal_test, levels_test)
    model_results["isotonic"] = metrics_iso

    results[model_name] = model_results

    print(f"\n{model_name}:")
    print(f"  Affine:   a={a_aff:.4f}, b={b_aff:.4f}")
    if popt_sig is not None:
        print(f"  Sigmoid:  a={popt_sig[0]:.4f}, b={popt_sig[1]:.4f}, c={popt_sig[2]:.4f}, d={popt_sig[3]:.4f}")
    for method in ["raw", "affine", "sigmoid", "isotonic"]:
        m = model_results.get(method, {})
        if "rmse" in m:
            print(f"  {method:10s} -> tau={m['kendall_tau']:.4f}, rho={m['spearman_rho']:.4f}, "
                  f"RMSE={m['rmse']:.4f}, CR={m['compression_ratio']:.4f}")


# ── Generate figure ───────────────────────────────────────────
fig, axes_arr = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

level_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
ideal_curve = 10.0 * (1.0 - level_vals)

for ax, (model_name, scores_arr) in zip(axes_arr, [("GPT-5 mini", gpt), ("Gemini 3 Flash", gem)]):
    mr = results[model_name]

    s_test_local = scores_arr[test_mask].astype(float)
    levels_test_local = levels[test_mask]

    # Compute calibrated test scores
    a_aff = mr["affine"]["params"]["a"]
    b_aff = mr["affine"]["params"]["b"]
    s_aff_local = affine(s_test_local, a_aff, b_aff)

    if "params" in mr.get("sigmoid", {}):
        sp = mr["sigmoid"]["params"]
        s_sig_local = sigmoid(s_test_local, sp["a"], sp["b"], sp["c"], sp["d"])
    else:
        s_sig_local = None

    # Isotonic
    iso_local = IsotonicRegression(increasing=True, out_of_bounds="clip")
    s_train_local = scores_arr[train_mask].astype(float)
    iso_local.fit(s_train_local, ideal[train_mask])
    s_iso_local = iso_local.predict(s_test_local)

    # Per-level means
    raw_means, aff_means, sig_means, iso_means = [], [], [], []
    for lv in level_vals:
        mask_lv = levels_test_local == lv
        raw_means.append(s_test_local[mask_lv].mean())
        aff_means.append(s_aff_local[mask_lv].mean())
        iso_means.append(s_iso_local[mask_lv].mean())
        if s_sig_local is not None:
            sig_means.append(s_sig_local[mask_lv].mean())

    # Plot — ideal, raw, then calibrations
    ax.plot(level_vals, ideal_curve, "k--", linewidth=2.5,
            label="Ideal ($S = 10(1-\\lambda)$)", zorder=5)
    ax.plot(level_vals, raw_means, "o-", color="#d62728", linewidth=2, markersize=8,
            label=f"Raw (RMSE={mr['raw']['rmse']:.2f})", zorder=4)
    ax.plot(level_vals, aff_means, "s-", color="#2ca02c", linewidth=2, markersize=8,
            label=f"Affine (RMSE={mr['affine']['rmse']:.2f})", zorder=3)
    if sig_means:
        ax.plot(level_vals, sig_means, "^-", color="#1f77b4", linewidth=2, markersize=8,
                label=f"Sigmoid (RMSE={mr['sigmoid']['rmse']:.2f})", zorder=2)
    ax.plot(level_vals, iso_means, "D-", color="#9467bd", linewidth=2, markersize=7,
            label=f"Isotonic (RMSE={mr['isotonic']['rmse']:.2f})", zorder=1)

    # Shade compression gap between raw and ideal
    ax.fill_between(level_vals, raw_means, ideal_curve, alpha=0.10, color="#d62728")

    ax.set_xlabel("Degradation level ($\\lambda$)", fontsize=13)
    ax.set_title(model_name, fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(0, 11)
    ax.set_xticks(level_vals)
    ax.legend(fontsize=9.5, loc="lower left")
    ax.grid(True, alpha=0.3)

axes_arr[0].set_ylabel("Score", fontsize=13)

fig.suptitle("Calibration Recovery: Raw vs. Calibrated Dose-Response Curves (Held-Out Articles)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIG_PATH}")

# ── Save results ──────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"Results saved to {RESULTS_PATH}")
