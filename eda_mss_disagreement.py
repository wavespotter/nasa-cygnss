"""
EDA: Explore factors contributing to Spotter–CYGNSS MSS and wind-speed disagreements.

Variables investigated:
    latitude, longitude, driftSpeed, significantWaveHeight,
    swellHeight10Seconds, CYGNSS_L2_V3.2_meanSquareSlopeUncertainty

Disagreement metrics:
    MSS  – two interchangeable fit options (set FIT_METHOD in Config):
        "power"  – power-law fit in log-log space:
                   CYGNSS_mss = a * Spotter_mss^b
        "tanh"   – hyperbolic-tangent saturation:
                   CYGNSS_mss = a * tanh(b * Spotter_mss)
                   asymptotes at 'a' for large Spotter MSS; tracks the
                   roll-off visible in the binned averages.
        Both fits are always plotted for comparison; mss_residual is
        computed against whichever fit is active.
    Wind speed – direct 1:1 difference (both instruments measure the same
    quantity on the same scale, so no fit is warranted):
        ws_diff = CYGNSS_ws - Spotter_ws
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import binned_statistic

# ── Config ──────────────────────────────────────────────────────────────────
# FIT_METHOD selects which curve is used to compute mss_residual:
#   "power"  – power law in log-log space: CYGNSS = a * Spotter^b
#   "tanh"   – hyperbolic-tangent saturation: CYGNSS = a * tanh(b * Spotter)
#              asymptotes at 'a' for large Spotter MSS, fitting the
#              roll-off visible in the binned averages.
FIT_METHOD = "power"   # "power" | "tanh"

SAT = "CYGNSS_L2_V3.2"
DATA_PATH = f"data/combined_spotter_{SAT}_2025121206_2026012906.csv"
PLOT_DIR = "plots/eda"
os.makedirs(PLOT_DIR, exist_ok=True)

EXPLORE_VARS = [
    "latitude", "longitude", "driftSpeed",
    "significantWaveHeight", "swellHeight10Seconds",
    f"{SAT}_meanSquareSlopeUncertainty",
]
EXPLORE_LABELS = [
    "Latitude [°]", "Longitude [°]", "Drift Speed [m/s]",
    "Sig. Wave Height [m]", "Swell Height >10 s [m]",
    "CYGNSS MSS Uncertainty [-]",
]

# ── Load & QAQC (mirrors driver_mss.py combined_qaqc, uncertainty_multiplier=4) ──
df = pd.read_csv(DATA_PATH, parse_dates=["time", "time_spotter"])
# if you don't filter for lower mss uncertainty, it's harder to see some trends as uncertainty is sometimes very high
df = df[df[f"{SAT}_meanSquareSlope"] > 4 * df[f"{SAT}_meanSquareSlopeUncertainty"]]
df = df[df["significantWaveHeight"] > 0.5]
df = df[df["windSpeed10Meter"] > 1.0]
df = df.reset_index(drop=True)

# ── MSS fits ────────────────────────────────────────────────────────────────
cyg_mss  = df[f"{SAT}_meanSquareSlope"].values
spot_mss = df["meanSquareSlopeExtended"].values
ok_mss   = (cyg_mss > 0) & (spot_mss > 0)

# Power-law fit: CYGNSS = a * Spotter^b  (log-log linear regression)
# Passes through the origin; a natural fit when both axes span decades.
mss_b_pow, mss_log_a = np.polyfit(np.log(spot_mss[ok_mss]), np.log(cyg_mss[ok_mss]), 1)
mss_a_pow  = np.exp(mss_log_a)
power_fit  = lambda x: mss_a_pow * x ** mss_b_pow          # noqa: E731
power_label = f"power: {mss_a_pow:.4f}·x^{mss_b_pow:.3f}"

# Tanh fit: CYGNSS = a * tanh(b * Spotter)
# 'a' is the asymptotic ceiling (saturates at high Spotter MSS);
# 'b' controls where the bend occurs.  Fit to raw data via curve_fit.
def _tanh_model(x, a, b):
    return a * np.tanh(b * x)

_p0 = [cyg_mss[ok_mss].max(), 1.0 / np.median(spot_mss[ok_mss])]
(mss_a_tanh, mss_b_tanh), _ = curve_fit(
    _tanh_model, spot_mss[ok_mss], cyg_mss[ok_mss], p0=_p0, maxfev=5000
)
tanh_fit   = lambda x: _tanh_model(x, mss_a_tanh, mss_b_tanh)  # noqa: E731
tanh_label = f"tanh: {mss_a_tanh:.4f}·tanh({mss_b_tanh:.2f}·x)"

# Select active fit for residuals
if FIT_METHOD == "tanh":
    mss_fit   = tanh_fit
    fit_label = tanh_label
else:  # default: power
    mss_fit   = power_fit
    fit_label = power_label

df["mss_residual"] = cyg_mss - mss_fit(spot_mss)

# Wind speed – direct 1:1 difference, no fit
cyg_ws  = df[f"{SAT}_windSpeed10Meter"].values
spot_ws = df["windSpeed10Meter"].values
df["ws_diff"] = cyg_ws - spot_ws

print(f"N = {len(df)} co-located observations after QAQC")
print(f"\nMSS fit ({FIT_METHOD}):  {fit_label}")
print(df[["mss_residual", "ws_diff"]].describe().round(5))

# ── Diagnostic: show the fits ────────────────────────────────────────────────
# Both fits are always drawn so you can compare them visually.
# The active fit (FIT_METHOD) is drawn solid+bold; the inactive one is dashed.
# MSS shown on both linear and log-log axes; wind speed on linear axes.
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax_mss_lin, ax_mss_log, ax_ws = axes

x_mss = np.linspace(spot_mss[ok_mss].min(), spot_mss[ok_mss].max(), 300)
x_ws  = np.linspace(spot_ws.min(), spot_ws.max(), 300)

def _fit_style(method, this):
    """Return (lw, ls, alpha) – bolder for the active fit."""
    return (2.5, "-", 1.0) if FIT_METHOD == this else (1.5, "--", 0.65)

for ax in (ax_mss_lin, ax_mss_log):
    ax.scatter(
        spot_mss if ax is ax_mss_lin else spot_mss[ok_mss],
        cyg_mss  if ax is ax_mss_lin else cyg_mss[ok_mss],
        s=4, alpha=0.25, color="steelblue",
    )
    lw, ls, al = _fit_style(FIT_METHOD, "power")
    ax.plot(x_mss, power_fit(x_mss), color="crimson",    lw=lw, ls=ls, alpha=al,
            label=power_label + (" ✓" if FIT_METHOD == "power" else ""))
    lw, ls, al = _fit_style(FIT_METHOD, "tanh")
    ax.plot(x_mss, tanh_fit(x_mss),  color="darkorange", lw=lw, ls=ls, alpha=al,
            label=tanh_label  + (" ✓" if FIT_METHOD == "tanh"  else ""))
    ax.plot(x_mss, x_mss, ls=":", color="k", alpha=0.4, lw=1, label="1 : 1")
    ax.set_xlabel("Spotter MSS (extended) [-]")
    ax.set_ylabel("CYGNSS MSS [-]")
    ax.legend(fontsize=8)

ax_mss_lin.set_title("MSS fits  (linear axes)")
ax_mss_log.set_xscale("log")
ax_mss_log.set_yscale("log")
ax_mss_log.set_title("MSS fits  (log-log axes)")

# Wind speed – 1:1 only, no fit
ax_ws.scatter(spot_ws, cyg_ws, s=4, alpha=0.25, color="steelblue")
ax_ws.plot(x_ws, x_ws, ls="--", color="k", alpha=0.4, lw=1, label="1 : 1")
ax_ws.set_xlabel("Spotter wind speed [m/s]")
ax_ws.set_ylabel("CYGNSS wind speed [m/s]")
ax_ws.set_title("Wind-speed fit  (linear axes)")
ax_ws.legend(fontsize=8)

plt.suptitle(f"MSS fits – active: {FIT_METHOD}  ({fit_label})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fit_diagnostic.png", dpi=200)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
corr_cols   = EXPLORE_VARS + ["mss_residual", "ws_diff"]
corr_labels = EXPLORE_LABELS + ["MSS residual (CYGNSS−fit) [-]", "WS diff (CYGNSS−Spotter) [m/s]"]

corr = df[corr_cols].corr()
corr.index   = corr_labels
corr.columns = corr_labels

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation matrix – MSS & wind-speed disagreement drivers")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png", dpi=200)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Scatter grid: each explore-variable vs a target disagreement metric
# ═══════════════════════════════════════════════════════════════════════════════
def scatter_grid(df, x_vars, x_labels, y_col, y_label, fname,
                 n_bins=18, min_count=10):
    ncols = 3
    nrows = int(np.ceil(len(x_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (xv, xl) in enumerate(zip(x_vars, x_labels)):
        ax = axes[i]
        x = df[xv].values
        y = df[y_col].values
        finite = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[finite], y[finite], s=4, alpha=0.25, color="steelblue")

        means, edges, _ = binned_statistic(x[finite], y[finite], statistic="mean", bins=n_bins)
        stds, _, _      = binned_statistic(x[finite], y[finite], statistic="std",  bins=n_bins)
        counts, _, _    = binned_statistic(x[finite], y[finite], statistic="count",bins=n_bins)
        ctrs = (edges[:-1] + edges[1:]) / 2
        valid = counts >= min_count
        ax.errorbar(ctrs[valid], means[valid], yerr=stds[valid],
                    fmt="o-", color="crimson", ms=4, lw=1.5,
                    elinewidth=1, capsize=3, label="Binned mean ± std")

        ax.axhline(0, ls="--", color="k", alpha=0.4)

        r = np.corrcoef(x[finite], y[finite])[0, 1]
        ax.set_xlabel(xl)
        ax.set_ylabel(y_label)
        ax.set_title(f"r = {r:.2f}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{y_label}  vs  contributing variables", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}", dpi=200)
    plt.show()


scatter_grid(df, EXPLORE_VARS, EXPLORE_LABELS, "mss_residual",
             "MSS residual (CYGNSS − fit) [-]", "scatter_mss_residual.png")
scatter_grid(df, EXPLORE_VARS, EXPLORE_LABELS, "ws_diff",
             "WS diff (CYGNSS − Spotter) [m/s]", "scatter_ws_diff.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Spatial map: lat/lon coloured by MSS and WS residuals
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in zip(
        axes,
        ["mss_residual", "ws_diff"],
        ["MSS residual (CYGNSS − fit) [-]", "WS diff (CYGNSS − Spotter) [m/s]"]):
    vabs = np.nanpercentile(np.abs(df[col]), 95)
    sc = ax.scatter(df["longitude"], df["latitude"],
                    c=df[col], s=8, cmap="RdBu_r",
                    vmin=-vabs, vmax=vabs, alpha=0.7)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(label)

plt.suptitle("Spatial distribution of MSS & wind-speed disagreements", fontsize=12)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/spatial_disagreement.png", dpi=200)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Time series: MSS ratio and wind-speed difference over time
# ═══════════════════════════════════════════════════════════════════════════════
df_s = df.sort_values("time")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.scatter(df_s["time"], df_s["mss_residual"], s=4, alpha=0.4, color="steelblue")
ax1.axhline(0, ls="--", color="k", alpha=0.5)
ax1.set_ylabel("MSS residual (CYGNSS − fit) [-]")
ax1.set_title("MSS residual over time")

ax2.scatter(df_s["time"], df_s["ws_diff"], s=4, alpha=0.4, color="darkorange")
ax2.axhline(0, ls="--", color="k", alpha=0.5)
ax2.set_ylabel("WS diff (CYGNSS − Spotter) [m/s]")
ax2.set_xlabel("Time")
ax2.set_title("Wind-speed difference (CYGNSS − Spotter) over time")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/timeseries_disagreement.png", dpi=200)
plt.show()

print(f"\nEDA complete. Plots saved to {PLOT_DIR}/")

