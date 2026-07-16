import pickle
import numpy as np
import scipy.constants as spc
import scipy.interpolate as spi
import matplotlib.pyplot as plt

plt.style.use("../eccm.mplstyle")

# Calculating relationships between array resolution, sensitivity, and time

dt = 10  # hour
sig_TB = 0.5  # K

# Planet dists
ground_dists = np.array([9.5, 19.2]) * spc.au  # m
body_rads = np.array([60268, 25559]) * 1e3  # m


fig = plt.figure(figsize=(11, 8))
fig2 = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(1, 1, 1)
tax = fig2.add_subplot(1, 1, 1)

# ALMA
# Tsys determined for ~1mm pwv
# Wavelengths from tech handbook
N = 50
f = np.array([39, 97.5, 145, 203, 233, 343])
wave = spc.c / (f * 1e9)
Tsys = np.array([52, 63, 68, 87, 87, 154])
Aeff = 0.72 * np.pi * 6**2  # m2
K = Aeff / (2 * spc.k)
SEFD = Tsys / K
sefd_spline = spi.CubicSpline(f, SEFD)
f = np.logspace(np.log10(min(f)), np.log10(max(f)), 100)
f = f[(f < 50) | (f > 80)]
wave = spc.c / (f * 1e9)
SEFD = sefd_spline(f)

BW = 7.5 * np.ones(len(f))  # GHz
# Extra stuff from tech handbook
sig_s = SEFD / 0.96 / 0.88 / np.sqrt(N * (N - 1) * 2 * BW * 1e9 * dt * 3600)
omega = sig_s / sig_TB * wave**2 / 2 / spc.k
ang_res = np.sqrt(4 * np.log(2) * omega / np.pi)
# ang_res = 3600 * np.degrees(ang_res)  # Arcseconds
res_dist = 2 * np.tan(ang_res[:, np.newaxis] / 2) * ground_dists
resolution = np.degrees(res_dist / body_rads)

ax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 0], label="ALMA")
tax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 1], label="ALMA")

# VLA
# SEFD from website
N = 27
data = np.loadtxt("vla_sefd_jy_ghz.csv", delimiter=",")
f = data[:, 0]
SEFD = data[:, 1] * 1e-26
sefd_spline = spi.CubicSpline(f, SEFD)
f = np.logspace(np.log10(min(f)), np.log10(max(f)), 100)
f = f[(f < 0.475) | (f > 1)]
wave = spc.c / (f * 1e9)
SEFD = sefd_spline(f)
BW = np.ones(len(f))
BW[(f <= 0.475)] = 0.2
BW[(f >= 1) & (f <= 2)] = 0.6
BW[(f >= 2) & (f <= 4)] = 1.5
BW[(f >= 4) & (f <= 8)] = 3.5
BW[(f >= 8) & (f <= 12)] = 3.5
BW[(f >= 12) & (f <= 18)] = 5.5
BW[(f >= 18)] = 7.5

# Extra stuff from tech handbook
sig_s = SEFD / np.sqrt(N * (N - 1) * 2 * BW * 1e9 * dt * 3600)
omega = sig_s / sig_TB * wave**2 / 2 / spc.k
ang_res = np.sqrt(4 * np.log(2) * omega / np.pi)
# ang_res = 3600 * np.degrees(ang_res)  # Arcseconds
res_dist = 2 * np.tan(ang_res[:, np.newaxis] / 2) * ground_dists
resolution = np.degrees(res_dist / body_rads)

ax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 0], label="VLA")
tax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 1], label="VLA")

# GMRT
# Aeff / Tsys * nant from Braun et al. 2019
N = 30
data = np.loadtxt("gmrt_array_ae_tsys_ghz_m2k.csv", delimiter=",")
f = data[:, 0]
AeoTs = data[:, 1] / 30
SEFD = 2 * spc.k / AeoTs
sefd_spline = spi.CubicSpline(f, SEFD)
f = np.logspace(np.log10(min(f)), np.log10(max(f)), 100)
wave = spc.c / (f * 1e9)
SEFD = sefd_spline(f)
BW = np.ones(len(f))
BW[(f <= 0.25)] = 0.05
BW[(f >= 0.25) & (f <= 0.5)] = 0.12
BW[(f >= 0.5) & (f <= 0.85)] = 0.2
BW[(f >= 0.85)] = 0.28
# Extra stuff from tech handbook
sig_s = SEFD / np.sqrt(N * (N - 1) * 2 * BW * 1e9 * dt * 3600)
omega = sig_s / sig_TB * wave**2 / 2 / spc.k
ang_res = np.sqrt(4 * np.log(2) * omega / np.pi)
# ang_res = 3600 * np.degrees(ang_res)  # Arcseconds
res_dist = 2 * np.tan(ang_res[:, np.newaxis] / 2) * ground_dists
resolution = np.degrees(res_dist / body_rads)

ax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 0], label="GMRT")
tax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 1], label="GMRT")


# SKA-mid
# Aeff / Tsys from Braun et al. 2019
N = 140
data = np.loadtxt("skamid_ae_tsys_ghz_m2K.csv", delimiter=",")
f = data[:, 0]
AeoTs = data[:, 1]
SEFD = 2 * spc.k / AeoTs
sefd_spline = spi.CubicSpline(f, SEFD)
f = np.logspace(np.log10(min(f)), np.log10(max(f)), 100)
wave = spc.c / (f * 1e9)
SEFD = sefd_spline(f)
BW = np.ones(len(f))
BW[(f <= 1.0)] = 0.5
BW[(f >= 1.0) & (f <= 4.0)] = 0.8
BW[(f >= 4.0) & (f <= 8.0)] = 3.0
BW[(f >= 8.0)] = 4.0
sig_s = SEFD / np.sqrt(N * (N - 1) * 2 * BW * 1e9 * dt * 3600)
omega = sig_s / sig_TB * wave**2 / 2 / spc.k
ang_res = np.sqrt(4 * np.log(2) * omega / np.pi)
# ang_res = 3600 * np.degrees(ang_res)  # Arcseconds
res_dist = 2 * np.tan(ang_res[:, np.newaxis] / 2) * ground_dists
resolution = np.degrees(res_dist / body_rads)

ax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 0], label="SKA")
tax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 1], label="SKA")


# ngVLA
N = 214  # 214 if mid included, 168 if not
etaQ = 0.9625
rec_data = pickle.load(open("receiver_data.pkl", "rb"), encoding="latin1")
keys = rec_data.keys()
f_list = []
Tsys_list = []
eta_A_list = []
bw_list = []
for k in keys:
    f_list.append(rec_data[k]["freq"])
    Tsys_list.append(rec_data[k]["tSys"])
    eta_A_list.append(rec_data[k]["eta_A"] * np.ones(len(rec_data[k]["freq"])))
    bw_list.append(rec_data[k]["max_bw"] * np.ones(len(rec_data[k]["freq"])))
f = np.concatenate(f_list)
Tsys = np.concatenate(Tsys_list)
eta_A = np.concatenate(eta_A_list)
BW = np.concatenate(bw_list)
D = 18
A = np.pi * (D / 2.0) ** 2
wave = spc.c / (f * 1e9)
SEFD = 2 * spc.k * Tsys / (etaQ * eta_A * A)
sig_s = SEFD / np.sqrt(N * (N - 1) * 2 * BW * 1e9 * dt * 3600)
omega = sig_s / sig_TB * wave**2 / 2 / spc.k
ang_res = np.sqrt(4 * np.log(2) * omega / np.pi)
# ang_res = 3600 * np.degrees(ang_res)  # Arcseconds
res_dist = 2 * np.tan(ang_res[:, np.newaxis] / 2) * ground_dists
resolution = np.degrees(res_dist / body_rads)

ax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 0], label="ngVLA")
tax.scatter(1e2 * spc.c / (f * 1e9), resolution[:, 1], label="ngVLA")


# Orbital radiometers

orbit_dists = np.array([1.1, 3])  # Scalar multiple of rad
antenna_diam = np.array([0.75, 4])
f = np.logspace(np.log10(0.3), np.log10(300), 100)
waves = spc.c / (f * 1e9)

ang_res = waves[:, np.newaxis] / antenna_diam
res_dist = 2 * np.tan(ang_res / 2) * (orbit_dists - 1)
resolution = np.degrees(res_dist)
ax.plot(1e2 * spc.c / (f * 1e9), resolution[:, 0], color="k", label="Juno-like")
ax.plot(
    1e2 * spc.c / (f * 1e9), resolution[:, 1], color="k", label="Cassini-like", ls="--"
)

tax.plot(1e2 * spc.c / (f * 1e9), resolution[:, 0], color="k", label="Juno-like")
tax.plot(
    1e2 * spc.c / (f * 1e9), resolution[:, 1], color="k", label="Cassini-like", ls="--"
)
ax.set_xlabel("Wavelength (cm)")
ax.set_ylabel("Planetographic spatial resolution (degrees)")
ax.set_title(
    r"Achievable resolution for $\tau=$10 hr, $\Delta T_B=$0.5 K" + "\n" + "Saturn"
)
ax.set_xlim(1e2 * spc.c / (0.3 * 1e9), 1e2 * spc.c / (300 * 1e9))
ax.set_ylim(1e-2, 1e3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(bbox_to_anchor=(1.0, 1.0))


tax.set_xlabel("Wavelength (cm)")
tax.set_ylabel("Planetographic spatial resolution (degrees)")
tax.set_title(
    r"Achievable resolution for $\tau=$10 hr, $\Delta T_B =$0.5 K" + "\n" + "Uranus"
)
tax.set_xlim(1e2 * spc.c / (0.3 * 1e9), 1e2 * spc.c / (300 * 1e9))
tax.set_ylim(1e-2, 1e3)
tax.set_xscale("log")
tax.set_yscale("log")
tax.legend(bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
fig2.tight_layout()

fig.savefig("gb_mwr_comparison_saturn.pdf")
fig2.savefig("gb_mwr_comparison_uranus.pdf")
