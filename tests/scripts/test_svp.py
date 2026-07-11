"""
Visual inspection of saturation vapor pressures and latent heats.

Panel 1: SVP vs temperature for all species (solid + liquid phases)
Panel 2: Latent heat vs temperature for all species (solid + liquid phases)
Panel 3: H2O SVP — pure vs NH3-H2O solution at various concentrations
Panel 4: NH3 SVP — pure vs NH3-H2O solution at various concentrations

Run: uv run python tests/scripts/test_svp.py
"""

import numpy as np
import matplotlib.pyplot as plt

from eccm_mrtm.eccm import thermo

Z = 1.0

SPECIES = {
    "H2O": {
        "solid": thermo.h2o_solid_saturation_vapor_pressure,
        "liquid": thermo.h2o_liquid_saturation_vapor_pressure,
        "triple": thermo.H2O_TRIPLE_POINT,
        "color": "tab:blue",
    },
    "NH3": {
        "solid": thermo.nh3_solid_saturation_vapor_pressure,
        "liquid": thermo.nh3_liquid_saturation_vapor_pressure,
        "triple": thermo.NH3_TRIPLE_POINT,
        "color": "tab:orange",
    },
    "H2S": {
        "solid": thermo.h2s_solid_saturation_vapor_pressure,
        "liquid": thermo.h2s_liquid_saturation_vapor_pressure,
        "triple": thermo.H2S_TRIPLE_POINT,
        "color": "tab:green",
    },
    "CH4": {
        "solid": thermo.ch4_solid_saturation_vapor_pressure,
        "liquid": thermo.ch4_liquid_saturation_vapor_pressure,
        "triple": thermo.CH4_TRIPLE_POINT,
        "color": "tab:red",
    },
    "PH3": {
        "solid": thermo.ph3_solid_saturation_vapor_pressure,
        "liquid": None,
        "triple": thermo.PH3_TRIPLE_POINT,
        "color": "tab:purple",
    },
}

T = np.linspace(50, 600, 1000)

# Compute SVP and latent heat for all species across both phases
svp_data = {}
lh_data = {}
for name, spec in SPECIES.items():
    tp = spec["triple"]
    svp_arr = np.full_like(T, np.nan)
    lh_arr = np.full_like(T, np.nan)

    for i, Ti in enumerate(T):
        if Ti < tp:
            s, l = spec["solid"](Ti, Z)
            svp_arr[i] = s
            lh_arr[i] = l
        else:
            if spec["liquid"] is not None:
                s, l = spec["liquid"](Ti, Z)
                svp_arr[i] = s
                lh_arr[i] = l
            else:
                s, l = spec["solid"](Ti, Z)
                svp_arr[i] = s
                lh_arr[i] = l

    svp_data[name] = svp_arr
    lh_data[name] = lh_arr

# Solution SVP comparison
concentrations = np.arange(0.0, 1.1, 0.1)
T_sol = np.linspace(180, 400, 500)

sol_h2o = {}
sol_nh3 = {}
for c in concentrations:
    h2o_svp = np.full_like(T_sol, np.nan)
    nh3_svp = np.full_like(T_sol, np.nan)
    Tf = thermo.h2o_nh3h2osolution_freezing_point(c)
    for i, Ti in enumerate(T_sol):
        if Ti >= Tf:
            h2o_svp[i] = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(Ti, c, Z)[
                0
            ]
            nh3_svp[i] = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(Ti, c, Z)[
                0
            ]
    sol_h2o[c] = h2o_svp
    sol_nh3[c] = nh3_svp

# Pure comparison curves
pure_h2o_svp = np.full_like(T_sol, np.nan)
pure_nh3_svp = np.full_like(T_sol, np.nan)
for i, Ti in enumerate(T_sol):
    if Ti < thermo.H2O_TRIPLE_POINT:
        pure_h2o_svp[i] = thermo.h2o_solid_saturation_vapor_pressure(Ti, Z)[0]
    else:
        pure_h2o_svp[i] = thermo.h2o_liquid_saturation_vapor_pressure(Ti, Z)[0]
    if Ti < thermo.NH3_TRIPLE_POINT:
        pure_nh3_svp[i] = thermo.nh3_solid_saturation_vapor_pressure(Ti, Z)[0]
    else:
        pure_nh3_svp[i] = thermo.nh3_liquid_saturation_vapor_pressure(Ti, Z)[0]

# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: SVP vs T
ax = axes[0, 0]
for name, spec in SPECIES.items():
    tp = spec["triple"]
    mask_solid = T < tp
    mask_liquid = T >= tp
    ax.semilogy(
        T[mask_solid],
        svp_data[name][mask_solid],
        "-",
        color=spec["color"],
        label=f"{name} (solid)",
    )
    if spec["liquid"] is not None:
        ax.semilogy(
            T[mask_liquid],
            svp_data[name][mask_liquid],
            "--",
            color=spec["color"],
            label=f"{name} (liquid)",
        )
    ax.axvline(tp, color=spec["color"], alpha=0.3, lw=0.8)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("SVP (bar)")
ax.set_title("Saturation Vapor Pressure")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(50, 600)
ax.grid(True, alpha=0.3)

# Panel 2: Latent heat vs T
ax = axes[0, 1]
for name, spec in SPECIES.items():
    tp = spec["triple"]
    mask_solid = T < tp
    mask_liquid = T >= tp
    ax.plot(
        T[mask_solid],
        lh_data[name][mask_solid] / 1e3,
        "-",
        color=spec["color"],
        label=f"{name} (solid)",
    )
    if spec["liquid"] is not None:
        ax.plot(
            T[mask_liquid],
            lh_data[name][mask_liquid] / 1e3,
            "--",
            color=spec["color"],
            label=f"{name} (liquid)",
        )
    ax.axvline(tp, color=spec["color"], alpha=0.3, lw=0.8)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Latent Heat (kJ/mol)")
ax.set_title("Latent Heat of Phase Transition")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(50, 600)
ax.grid(True, alpha=0.3)

# Panel 3: H2O pure vs solution
ax = axes[1, 0]
ax.semilogy(T_sol, pure_h2o_svp, "k-", lw=2, label="Pure H2O")
cmap = plt.cm.coolwarm
for j, c in enumerate(concentrations):
    if c == 0.0 or c == 1.0:
        continue
    color = cmap(c)
    mask = ~np.isnan(sol_h2o[c])
    if mask.any():
        ax.semilogy(
            T_sol[mask],
            sol_h2o[c][mask],
            "-",
            color=color,
            alpha=0.7,
            label=f"c={c:.1f}",
        )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("SVP (bar)")
ax.set_title("H2O SVP: Pure vs NH3-H2O Solution")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(180, 400)
ax.grid(True, alpha=0.3)

# Panel 4: NH3 pure vs solution
ax = axes[1, 1]
ax.semilogy(T_sol, pure_nh3_svp, "k-", lw=2, label="Pure NH3")
for j, c in enumerate(concentrations):
    if c == 0.0 or c == 1.0:
        continue
    color = cmap(c)
    mask = ~np.isnan(sol_nh3[c])
    if mask.any():
        ax.semilogy(
            T_sol[mask],
            sol_nh3[c][mask],
            "-",
            color=color,
            alpha=0.7,
            label=f"c={c:.1f}",
        )
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("SVP (bar)")
ax.set_title("NH3 SVP: Pure vs NH3-H2O Solution")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(180, 400)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
