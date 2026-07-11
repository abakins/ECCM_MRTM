"""
Visual inspection of the Helmholtz EOS: compressibility Z and heat capacity Cp.

Plots Z*R (effective gas constant) and Cp as functions of T and P for
various gas mixtures relevant to giant planet atmospheres.

Solid lines = real-gas EOS result
Dashed lines = ideal-gas value (Z=1 for gas constant, Cp0 for heat capacity)

Run: python tests/scripts/test_eos.py

Note: Warnings will print for conditions outside EOS validity (high P / low T).
These are expected — the EOS falls back to ideal gas in those regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eccm_mrtm" / "eccm"))
import eos  # noqa: E402

compute_Z = eos.compute_Z
compute_Cp = eos.compute_Cp
ideal_alpha_dtau_dtau_cp = eos.ideal_alpha_dtau_dtau_cp
ideal_alpha_dtau_dtau_coef = eos.ideal_alpha_dtau_dtau_coef
H2_NI = eos.H2_NI
H2_TC = eos.H2_TC
H2_VI = eos.H2_VI
H2_UI = eos.H2_UI
HE_NI = eos.HE_NI
HE_TC = eos.HE_TC
HE_VI = eos.HE_VI
HE_UI = eos.HE_UI
CH4_NI = eos.CH4_NI
CH4_TC = eos.CH4_TC
CH4_VI = eos.CH4_VI
CH4_UI = eos.CH4_UI
H2O_TC = eos.H2O_TC
H2O_IDEAL_N = eos.H2O_IDEAL_N
H2O_IDEAL_GAMMA = eos.H2O_IDEAL_GAMMA
ORTHO_H2_NI = eos.ORTHO_H2_NI
ORTHO_H2_TC = eos.ORTHO_H2_TC
ORTHO_H2_VI = eos.ORTHO_H2_VI
ORTHO_H2_UI = eos.ORTHO_H2_UI
PARA_H2_NI = eos.PARA_H2_NI
PARA_H2_TC = eos.PARA_H2_TC
PARA_H2_VI = eos.PARA_H2_VI
PARA_H2_UI = eos.PARA_H2_UI

R = 8.314462618  # J/(mol*K)

# ==============================================================================
# Setup: Temperature and Pressure grids
# ==============================================================================
T_range = np.linspace(50, 800, 200)
P_bars = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
P_range = P_bars * 1e5  # Pa

T_for_P_sweep = np.array([100.0, 200.0, 300.0, 500.0])
P_sweep = np.logspace(np.log10(0.1), np.log10(1000), 100) * 1e5


# ==============================================================================
# Ideal gas Cp calculation (for dashed reference lines)
# ==============================================================================
def compute_ideal_Cp_over_R(T_arr, x_h2, x_he, x_ch4, x_h2o, h2_type):
    """Compute ideal-gas Cp/R as a function of T for a given composition."""
    result = np.zeros_like(T_arr)
    for i, T in enumerate(T_arr):
        cp0 = 0.0
        if x_h2 > 0:
            if h2_type == 1:
                cp0 += x_h2 * ideal_alpha_dtau_dtau_cp(ORTHO_H2_NI, ORTHO_H2_TC, ORTHO_H2_VI, ORTHO_H2_UI, T)
            elif h2_type == 2:
                cp0 += x_h2 * ideal_alpha_dtau_dtau_cp(PARA_H2_NI, PARA_H2_TC, PARA_H2_VI, PARA_H2_UI, T)
            else:
                cp0 += x_h2 * ideal_alpha_dtau_dtau_cp(H2_NI, H2_TC, H2_VI, H2_UI, T)
        if x_he > 0:
            cp0 += x_he * ideal_alpha_dtau_dtau_cp(HE_NI, HE_TC, HE_VI, HE_UI, T)
        if x_ch4 > 0:
            cp0 += x_ch4 * ideal_alpha_dtau_dtau_cp(CH4_NI, CH4_TC, CH4_VI, CH4_UI, T)
        if x_h2o > 0:
            tau_h2o = H2O_TC / T
            alpha0_tt = ideal_alpha_dtau_dtau_coef(tau_h2o, H2O_IDEAL_N, H2O_IDEAL_GAMMA)
            cp0 += x_h2o * (-tau_h2o**2 * alpha0_tt + 1.0)
        result[i] = cp0
    return result


# ==============================================================================
# Compositions
# ==============================================================================
compositions = {
    "Pure H2 (normal)": (1.0, 0.0, 0.0, 0.0, 0),
    "Pure He": (0.0, 1.0, 0.0, 0.0, 0),
    "Jupiter bulk (86/14)": (0.86, 0.14, 0.0, 0.0, 0),
    "H2/He + 5% CH4": (0.81, 0.14, 0.05, 0.0, 0),
    "H2/He + 5% H2O": (0.81, 0.14, 0.0, 0.05, 0),
    "H2/He + 3% CH4 + 2% H2O": (0.81, 0.14, 0.03, 0.02, 0),
}

h2_variants = {
    "Normal H2": 0,
    "Ortho H2": 1,
    "Para H2": 2,
}

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def compute_grid_Z(T_arr, P, x_h2, x_he, x_ch4, x_h2o):
    return np.array([compute_Z(P, T, x_h2, x_he, x_ch4, x_h2o) for T in T_arr])


def compute_grid_Cp(T_arr, P, x_h2, x_he, x_ch4, x_h2o, h2_type):
    return np.array([compute_Cp(P, T, x_h2, x_he, x_ch4, x_h2o, h2_type) for T in T_arr])


# ==============================================================================
# Figure 1: Z*R (effective gas constant) vs Temperature at fixed pressures
# ==============================================================================
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
axes1 = axes1.flatten()

for idx, (name, (x_h2, x_he, x_ch4, x_h2o, h2t)) in enumerate(compositions.items()):
    ax = axes1[idx]
    ax.axhline(R, color="k", ls="--", lw=1.2, alpha=0.7, label="Ideal (Z=1)")
    for i, P in enumerate(P_range):
        Z_arr = compute_grid_Z(T_range, P, x_h2, x_he, x_ch4, x_h2o)
        ZR = Z_arr * R
        ax.plot(T_range, ZR, color=colors[i % len(colors)], label=f"{P/1e5:.1f} bar")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Z*R (J/mol/K)")
    ax.legend(fontsize=7, loc="upper right")

fig1.suptitle("Effective Gas Constant Z*R vs Temperature\n(dashed = ideal gas)", fontsize=13)
fig1.tight_layout()


# ==============================================================================
# Figure 2: Z*R vs Pressure at fixed temperatures (log-P axis)
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
axes2 = axes2.flatten()

for idx, (name, (x_h2, x_he, x_ch4, x_h2o, h2t)) in enumerate(compositions.items()):
    ax = axes2[idx]
    ax.axhline(R, color="k", ls="--", lw=1.2, alpha=0.7, label="Ideal (Z=1)")
    for i, T in enumerate(T_for_P_sweep):
        Z_arr = np.array([compute_Z(P, T, x_h2, x_he, x_ch4, x_h2o) for P in P_sweep])
        ZR = Z_arr * R
        ax.plot(P_sweep / 1e5, ZR, color=colors[i % len(colors)], label=f"T={T:.0f} K")
    ax.set_xscale("log")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("P (bar)")
    ax.set_ylabel("Z*R (J/mol/K)")
    ax.legend(fontsize=8)

fig2.suptitle("Effective Gas Constant Z*R vs Pressure\n(dashed = ideal gas)", fontsize=13)
fig2.tight_layout()


# ==============================================================================
# Figure 3: Cp vs Temperature at fixed pressures
# ==============================================================================
fig3, axes3 = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
axes3 = axes3.flatten()

for idx, (name, (x_h2, x_he, x_ch4, x_h2o, h2t)) in enumerate(compositions.items()):
    ax = axes3[idx]
    Cp_ideal = compute_ideal_Cp_over_R(T_range, x_h2, x_he, x_ch4, x_h2o, h2t)
    ax.plot(T_range, Cp_ideal, "k--", lw=1.2, alpha=0.7, label="Ideal Cp")
    for i, P in enumerate(P_range):
        Cp_arr = compute_grid_Cp(T_range, P, x_h2, x_he, x_ch4, x_h2o, h2t)
        ax.plot(T_range, Cp_arr / R, color=colors[i % len(colors)], label=f"{P/1e5:.1f} bar")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Cp/R")
    ax.legend(fontsize=7, loc="upper right")

fig3.suptitle("Isobaric Heat Capacity Cp/R vs Temperature\n(dashed = ideal gas)", fontsize=13)
fig3.tight_layout()


# ==============================================================================
# Figure 4: Cp vs Pressure at fixed temperatures (log-P axis)
# ==============================================================================
fig4, axes4 = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
axes4 = axes4.flatten()

for idx, (name, (x_h2, x_he, x_ch4, x_h2o, h2t)) in enumerate(compositions.items()):
    ax = axes4[idx]
    for i, T in enumerate(T_for_P_sweep):
        Cp_ideal_val = compute_ideal_Cp_over_R(np.array([T]), x_h2, x_he, x_ch4, x_h2o, h2t)[0]
        ax.axhline(Cp_ideal_val, color=colors[i % len(colors)], ls="--", lw=1.0, alpha=0.5)
        Cp_arr = np.array([compute_Cp(P, T, x_h2, x_he, x_ch4, x_h2o, h2t) for P in P_sweep])
        ax.plot(P_sweep / 1e5, Cp_arr / R, color=colors[i % len(colors)], label=f"T={T:.0f} K")
    ax.set_xscale("log")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("P (bar)")
    ax.set_ylabel("Cp/R")
    ax.legend(fontsize=8)

fig4.suptitle("Isobaric Heat Capacity Cp/R vs Pressure\n(dashed = ideal gas at same T)", fontsize=13)
fig4.tight_layout()


# ==============================================================================
# Figure 5: H2 variant comparison (normal vs ortho vs para)
# ==============================================================================
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, h2t) in enumerate(h2_variants.items()):
    Cp_ideal = compute_ideal_Cp_over_R(T_range, 1.0, 0.0, 0.0, 0.0, h2t)
    ax5a.plot(T_range, Cp_ideal, ls="--", color=colors[i], alpha=0.6)
    Cp_1bar = compute_grid_Cp(T_range, 1e5, 1.0, 0.0, 0.0, 0.0, h2t)
    ax5a.plot(T_range, Cp_1bar / R, color=colors[i], label=f"{name} (1 bar)")
    Cp_100bar = compute_grid_Cp(T_range, 100e5, 1.0, 0.0, 0.0, 0.0, h2t)
    ax5a.plot(T_range, Cp_100bar / R, color=colors[i], ls="-.", lw=1.5, label=f"{name} (100 bar)")

ax5a.set_xlabel("T (K)")
ax5a.set_ylabel("Cp/R")
ax5a.set_title("Pure H2 variants (dashed = ideal)")
ax5a.legend(fontsize=7)

# Z comparison for H2 variants (Z is identical since residual params differ only slightly)
ax5b.axhline(1.0, color="k", ls="--", lw=1.2, alpha=0.7, label="Ideal (Z=1)")
for i, (name, h2t) in enumerate(h2_variants.items()):
    Z_arr = compute_grid_Z(T_range, 100e5, 1.0, 0.0, 0.0, 0.0)
    ax5b.plot(T_range, Z_arr, color=colors[i], label=f"{name} (100 bar)")
ax5b.set_xlabel("T (K)")
ax5b.set_ylabel("Z")
ax5b.set_title("Compressibility Z for pure H2 at 100 bar")
ax5b.legend(fontsize=8)

fig5.suptitle("H2 Variant Comparison: Normal vs Ortho vs Para\n(dashed = ideal gas)", fontsize=13)
fig5.tight_layout()


# ==============================================================================
# Figure 6: Effect of CH4 and H2O mole fraction on Cp and Z
# ==============================================================================
fig6, axes6 = plt.subplots(2, 2, figsize=(11, 9))

x_ch4_sweep = np.linspace(0, 0.10, 6)
x_h2o_sweep = np.linspace(0, 0.10, 6)
P_fixed = 50e5  # 50 bar

# Cp vs CH4 fraction
ax = axes6[0, 0]
for i, x_ch4 in enumerate(x_ch4_sweep):
    x_h2 = 0.86 - x_ch4
    Cp_ideal = compute_ideal_Cp_over_R(T_range, x_h2, 0.14, x_ch4, 0.0, 0)
    ax.plot(T_range, Cp_ideal, ls="--", color=colors[i % len(colors)], alpha=0.5)
    Cp_arr = compute_grid_Cp(T_range, P_fixed, x_h2, 0.14, x_ch4, 0.0, 0)
    ax.plot(T_range, Cp_arr / R, color=colors[i % len(colors)], label=f"CH4={x_ch4:.2f}")
ax.set_xlabel("T (K)")
ax.set_ylabel("Cp/R")
ax.set_title(f"Cp/R vs T at {P_fixed/1e5:.0f} bar, varying CH4\n(dashed = ideal)")
ax.legend(fontsize=7)

# Z vs CH4 fraction
ax = axes6[0, 1]
ax.axhline(1.0, color="k", ls="--", lw=1.2, alpha=0.7)
for i, x_ch4 in enumerate(x_ch4_sweep):
    x_h2 = 0.86 - x_ch4
    Z_arr = compute_grid_Z(T_range, P_fixed, x_h2, 0.14, x_ch4, 0.0)
    ax.plot(T_range, Z_arr, color=colors[i % len(colors)], label=f"CH4={x_ch4:.2f}")
ax.set_xlabel("T (K)")
ax.set_ylabel("Z")
ax.set_title(f"Z vs T at {P_fixed/1e5:.0f} bar, varying CH4\n(dashed = ideal)")
ax.legend(fontsize=7)

# Cp vs H2O fraction
ax = axes6[1, 0]
for i, x_h2o in enumerate(x_h2o_sweep):
    x_h2 = 0.86 - x_h2o
    Cp_ideal = compute_ideal_Cp_over_R(T_range, x_h2, 0.14, 0.0, x_h2o, 0)
    ax.plot(T_range, Cp_ideal, ls="--", color=colors[i % len(colors)], alpha=0.5)
    Cp_arr = compute_grid_Cp(T_range, P_fixed, x_h2, 0.14, 0.0, x_h2o, 0)
    ax.plot(T_range, Cp_arr / R, color=colors[i % len(colors)], label=f"H2O={x_h2o:.2f}")
ax.set_xlabel("T (K)")
ax.set_ylabel("Cp/R")
ax.set_title(f"Cp/R vs T at {P_fixed/1e5:.0f} bar, varying H2O\n(dashed = ideal)")
ax.legend(fontsize=7)

# Z vs H2O fraction
ax = axes6[1, 1]
ax.axhline(1.0, color="k", ls="--", lw=1.2, alpha=0.7)
for i, x_h2o in enumerate(x_h2o_sweep):
    x_h2 = 0.86 - x_h2o
    Z_arr = compute_grid_Z(T_range, P_fixed, x_h2, 0.14, 0.0, x_h2o)
    ax.plot(T_range, Z_arr, color=colors[i % len(colors)], label=f"H2O={x_h2o:.2f}")
ax.set_xlabel("T (K)")
ax.set_ylabel("Z")
ax.set_title(f"Z vs T at {P_fixed/1e5:.0f} bar, varying H2O\n(dashed = ideal)")
ax.legend(fontsize=7)

fig6.suptitle("Effect of CH4 and H2O Mole Fraction on Cp and Z", fontsize=13)
fig6.tight_layout()

plt.show()
