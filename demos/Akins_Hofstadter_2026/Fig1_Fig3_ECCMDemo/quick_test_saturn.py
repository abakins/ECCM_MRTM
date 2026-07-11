import os
import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors


os.environ["NUMBA_DISABLE_JIT"] = "1"
print("RENABLE JIT!!")


from eccm_mrtm import eccm, mrtm

# p_grid = np.logspace(2, 9, 10000)
p_grid = np.logspace(2, np.log10(10**1.4 * spc.bar), 1000)
saturn_gravity = 10.44  # m/s2


# Load RO profile
data_dir = eccm.__path__[0]
saturn_ro = np.loadtxt(
    os.path.join(data_dir, "data/ro_profiles/saturn_cassini_schinder2011.txt"),
    skiprows=3,
    delimiter=",",
)
saturn_pressure = saturn_ro[:, 0] * 1e2  # Pa
saturn_temperature = saturn_ro[:, 1]

gases1 = eccm.solar_concentration()

gases = [  # eccm.GasInput('H2O', deep=10*gases1['O'], rh=0.7),
    eccm.GasInput("H2O", deep=0.0107, rh=0.7),
    eccm.GasInput("NH3", deep=3 * gases1["N"], rh=0.7),
    eccm.GasInput("H2S", deep=10 * gases1["S"], rh=1.0),
    eccm.GasInput("CH4", deep=10 * gases1["C"], rh=1.0),
]

result = eccm.run_eccm(
    p_grid,
    saturn_pressure,
    saturn_temperature,
    saturn_gravity,
    gases=gases,
    bulk_h2=0.88,
    bulk_he=0.12,
    latent_heat_update=True,
    force_reference_above_pressure=1.0 * spc.bar,
)

pressure_grid = result["pressure"]
temperature_grid = result["temperature"]
altitude_grid = result["altitude"]
x_h2o = result["gas_profiles"]["H2O"]
x_nh3 = result["gas_profiles"]["NH3"]
x_h2s = result["gas_profiles"]["H2S"]
x_ch4 = result["gas_profiles"]["CH4"]

a_h2osolid = result["aerosol_densities"]["H2O"]["solid"]
a_h2oliquid = result["aerosol_densities"]["H2O"]["liquid"]
a_nh3solid = result["aerosol_densities"]["NH3"]["solid"]
a_nh3liquid = result["aerosol_densities"]["NH3"]["liquid"]
a_h2ssolid = result["aerosol_densities"]["H2S"]["solid"]
a_h2sliquid = result["aerosol_densities"]["H2S"]["liquid"]
a_ch4solid = result["aerosol_densities"]["CH4"]["solid"]
a_ch4liquid = result["aerosol_densities"]["CH4"]["liquid"]
a_h2osolution = result["aerosol_densities"]["H2O_NH3_SOLUTION"]["liquid"]
a_nh4sh = result["aerosol_densities"]["NH4SH"]["solid"]


def p_to_z(x):
    return np.interp(x, np.log10(pressure_grid / spc.bar)[::-1], altitude_grid[::-1])


def z_to_p(x):
    return np.interp(x, altitude_grid[::-1], np.log10(pressure_grid / spc.bar)[::-1])


gs = GridSpec(1, 3, width_ratios=[2, 1, 1], height_ratios=[1])


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(gs[1])
ax.plot(temperature_grid, np.log10(pressure_grid / spc.bar), color="k")
ax.set_xlabel("Temperature (K)", labelpad=12)
ax.set_ylim(-1, 3)
ax.set_xlim(50, 1000)
ax.invert_yaxis()
ax.yaxis.set_tick_params(labelcolor="none")
ax.tick_params(right=False, which="both")
secax = ax.secondary_yaxis("right", functions=(p_to_z, z_to_p))
secax.set_yticks([-250, -100, 0, 100, 250, 500, 1000])
secax.tick_params(right=False, which="minor")
secax.yaxis.set_tick_params(labelcolor="none")

ax = fig.add_subplot(gs[0])
ax.plot(x_h2o, np.log10(pressure_grid / spc.bar), label="H$_2$O", color="b")
ax.plot(x_nh3, np.log10(pressure_grid / spc.bar), label="NH$_3$", color="r")
ax.plot(x_h2s, np.log10(pressure_grid / spc.bar), label="H$_2$S", color="g")
ax.plot(x_ch4, np.log10(pressure_grid / spc.bar), label="CH$_4$", color="gray")
# ax.plot(x_ph3, np.log10(pressure_grid / spc.bar), label='PH$_3$', color='purple')
ax.set_xlabel("Gas mole fraction")
ax.set_ylabel(r"$\log_{10}$(Pressure) (bar)")
ax.set_xscale("log")
ax.set_xlim(1e-9, 1e-1)
ax.set_ylim(-1, 3)
ax.invert_yaxis()
ax.tick_params(right=False, which="both")
secax = ax.secondary_yaxis("right", functions=(p_to_z, z_to_p))
secax.set_yticks([-250, -100, 0, 100, 250, 500, 1000])
secax.tick_params(right=False, which="minor")
secax.yaxis.set_tick_params(labelcolor="none")
ax.text(3e-9, 2.5, "H$_2$O", color="b")
ax.text(3e-9, 2.25, "NH$_3$", color="r")
ax.text(3e-9, 2.0, "H$_2$S", color="g")
ax.text(3e-9, 1.75, "CH$_4$", color="gray")
# ax.text(3e-9, 2.5, 'PH$_3$', color='b')


paired_cmap = plt.get_cmap("Paired")

ax = fig.add_subplot(gs[2])
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_h2osolid,
    label="H$_2$O solid",
    color=paired_cmap(0),
)
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_h2oliquid,
    label="H$_2$O liquid",
    color=paired_cmap(1),
)
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_h2osolution,
    label="H$_2$O solution",
    color="turquoise",
)
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_nh3solid,
    label="NH$_3$ solid",
    color=paired_cmap(4),
)
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_nh3liquid,
    label="NH$_3$ liquid",
    color=paired_cmap(5),
)
# ax.fill_betweenx(np.log10(pressure_grid / spc.bar), 0., a_h2ssolid, label='H$_2$S solid', color=paired_cmap(2))
# ax.fill_betweenx(np.log10(pressure_grid / spc.bar), 0., a_h2sliquid, label='H$_2$S liquid', color=paired_cmap(3))
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_ch4solid,
    label="CH$_4$ solid",
    color="gray",
)
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar),
    0.0,
    a_ch4liquid,
    label="CH$_4$ liquid",
    color="k",
)
# ax.fill_betweenx(np.log10(pressure_grid / spc.bar), 0., a_ph3solid, label='PH$_3$ solid', color='purple')
ax.fill_betweenx(
    np.log10(pressure_grid / spc.bar), 0.0, a_nh4sh, label="NH4SH", color="orange"
)
ax.set_xlabel("Cloud layers")
ax.set_xscale("log")
ax.set_ylim(-1, 3)
ax.set_xlim(1e-6, 1000)
ax.xaxis.set_tick_params(labelcolor="none")
ax.invert_yaxis()
ax.tick_params(bottom=False, right=False, which="both")
ax.yaxis.set_tick_params(labelcolor="none")
ax.text(3e-6, 2.75, "H$_2$O solution", color="turquoise")
ax.text(3e-6, 2.5, "H$_2$O", color=paired_cmap(0))
ax.text(3e-6, 2.25, "NH$_4$SH", color="orange")
ax.text(3e-6, 2.0, "NH$_3$", color=paired_cmap(4))
ax.text(3e-6, 1.75, "CH$_4$", color="gray")


secax = ax.secondary_yaxis("right", functions=(p_to_z, z_to_p))
secax.yaxis.set_major_formatter(ticker.ScalarFormatter())
secax.set_yticks([-250, -100, 0, 100, 250, 500, 800])
secax.tick_params(right=False, which="minor")
secax.set_ylabel("Altitude (km)", rotation=-90, labelpad=20)

fig.suptitle("Saturn", y=0.92, fontsize="x-large")
fig.tight_layout(w_pad=-2)
fig.savefig("saturn_eccm.png", dpi=300)
