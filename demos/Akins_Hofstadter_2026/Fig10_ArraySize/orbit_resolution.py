import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
import scipy.constants as spc 
from tqdm.auto import tqdm
from matplotlib.lines import Line2D
import seaborn as sns 
plt.style.use('/Users/akins/Documents/Research/Akins.mplstyle')


# Uranus atmosphere resolution 
# X - Wavelength 
# Y - Aperture size 
# Color - Orbit distance 
# Style - Latitude resolution 

rad = 25559  # km
frequency = np.logspace(np.log10(300e6), np.log10(30e9), 15)
wave = spc.c / frequency
dist = np.array([1.2, 2, 3]) * rad
resolution = np.array([2, 5, 33])  # deg 
res_dist = np.radians(resolution) * rad  # km
inst_angle = np.degrees(2 * np.arctan(res_dist / 2 / (dist[:, np.newaxis] - rad)))
inst_diam = 0.88 * wave[:, np.newaxis, np.newaxis] / np.sin(np.radians(inst_angle))  # m 
dray = xr.DataArray(inst_diam, coords={'wavelength': wave, 'distance': dist/rad, 'resolution': resolution}, name='width') 
dframe = dray.to_dataframe().reset_index() 
dframe['distance'] = np.round(dframe['distance'], decimals=1)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
colors = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
colors = [colors(0), colors(int(0.5 * 256)), colors(256)]
dframe = dframe.rename(columns={'wavelength': 'Wavelength (m)', 'distance': 'Orbiter distance (R$_P$)', 
                                'resolution': 'Latitude res. (deg.)', 'width': 'Aperture width (m)'})
sns.lineplot(data=dframe, x='Wavelength (m)', y='Aperture width (m)', style='Orbiter distance (R$_P$)', 
             hue='Latitude res. (deg.)', size='Latitude res. (deg.)', ax=ax, 
             palette=sns.color_palette('muted'), markers=['D', 's', 'o'], 
             dashes=False, zorder=0, markeredgecolor='k', sizes = [2, 4, 6])
ax.set_yscale('log')
ax.set_ylim(1e-2, 5)
ax.set_xscale('log')
ax.invert_xaxis()

ylim = ax.get_ylim()
xlim = ax.get_xlim()
juno_wave = np.array([0.5, 0.24, 0.1155, 0.0575, 0.03, 0.0137])
juno_diams = np.array([1.6, 0.77, 0.72, 0.36, 0.18, 0.15])
# plt.vlines(juno_wave, *ylim, color='k')
ax.hlines(3.7, *xlim, color='r', ls='-', linewidth=2, zorder=1)
ax.scatter(juno_wave, juno_diams, c='k', s=100, marker='x', zorder=2, linewidth=2)

# Annotations
for i in range(len(juno_wave)):
    if i == 0:
        alpha = 0.8
    else: 
        alpha = 0.
    ax.annotate('Juno Microwave\nRadiometer Antennas', 
                xy=(juno_wave[i], juno_diams[i]), 
                xytext=(juno_wave[1] * 2, juno_diams[1] * 1e-1), 
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, ), 
                fontsize=14, ha='center', va='bottom', alpha=alpha,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=alpha))

ax.annotate('Spacecraft\nHigh-Gain Antenna', 
            xy=(1.8e-2, 3.7), 
            xytext=(1.8e-2, 1.8), 
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), 
            fontsize=12, ha='center', va='bottom', color='red', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

msd = {'D': 5, 'o': 10, 's': 7}
for line in ax.lines:
    try: 
        gm = str(line.get_marker())
        line.set_markersize(msd[gm])
    except KeyError:
        pass

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig('antennasize.png', dpi=300)


# Uranus atmosphere resolution 
# Zoom for close approaches 
# X - Orbit distance
# Y - Aperture size 
# Color, Juno channels 
# Plot, Target latitude res - 2, 5, 

frequency = np.array([600e6, 1.25e9, 2.6e9, 5.2e9, 10e9, 22e9])
wave = 3e8 / frequency
dist = (1 + np.linspace(0.1, 0.75, 100)) * rad
resolution = np.array([2, 5])  # deg 
res_dist = np.radians(resolution) * rad  # km

inst_angle = np.degrees(2 * np.arctan(res_dist / 2 / (dist[:, np.newaxis] - rad)))
inst_diam = 0.88 * wave[:, np.newaxis, np.newaxis] / np.sin(np.radians(inst_angle))  # m 
dray = xr.DataArray(inst_diam, coords={'wavelength': wave, 'distance': dist/rad, 'resolution': resolution}, name='width') 
dframe = dray.to_dataframe().reset_index() 
dframe['wavelength'] = np.round(dframe['wavelength']*1e2, decimals=1)

fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 2, 1)
colors = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
colors = [colors(0), colors(int(0.5 * 256)), colors(256)]
dframe = dframe.rename(columns={'wavelength': 'Wavelength (cm)', 'distance': 'Orbiter distance (R$_P$)', 
                                'resolution': 'Latitude resolution (deg.)', 'width': 'Aperture width (m)'})

udframe = dframe.loc[dframe['Latitude resolution (deg.)'] == 2]

sns.lineplot(data=udframe, y='Aperture width (m)', x='Orbiter distance (R$_P$)', 
             hue='Wavelength (cm)', ax=ax, palette=sns.color_palette('muted'))
ax.set_yscale('log')
ax.set_title('Latitude resolution: 2 deg')
ax.set_ylim(1e-2, 1e1)
ax.get_legend().remove()

ax = fig.add_subplot(1, 2, 2)
udframe = dframe.loc[dframe['Latitude resolution (deg.)'] == 5]
sns.lineplot(data=udframe, y='Aperture width (m)', x='Orbiter distance (R$_P$)', 
             hue='Wavelength (cm)', ax=ax, palette=sns.color_palette('muted'))
ax.set_yscale('log')
ax.set_title('Latitude resolution: 5 deg')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_legend().set_title('Wavelength (cm)')
ax.set_ylim(1e-2, 1e1)
fig.tight_layout()
fig.savefig('latres_width.png', dpi=300)
# Uranus atmosphere resolution 
# HGA - 3.7 m 
# X - Orbit distance
# Y - Latitude resolution
# Color, Juno channels 


frequency = np.array([600e6, 1.25e9, 2.6e9, 5.2e9, 10e9, 22e9])
wave = 3e8 / frequency
dist = (1 + np.linspace(0.1, 4, 1000)) * rad
inst_diam = 3.7 

inst_angle = np.arcsin(0.88 * wave[:, np.newaxis, np.newaxis] / inst_diam)
res_dist = 2 * np.tan(inst_angle / 2) * (dist[:, np.newaxis] - rad)
resolution = np.degrees(res_dist / rad)

dray = xr.DataArray(resolution.squeeze(), coords={'wavelength': wave, 'distance': dist/rad}, name='resolution') 
dframe = dray.to_dataframe().reset_index() 
dframe['wavelength'] = np.round(dframe['wavelength']*1e2, decimals=1)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
colors = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
colors = [colors(0), colors(int(0.5 * 256)), colors(256)]
dframe = dframe.rename(columns={'wavelength': 'Wavelength (cm)', 'distance': 'Orbiter distance (R$_P$)', 
                                'resolution': 'Latitude resolution (deg.)'})

sns.lineplot(data=dframe, y='Latitude resolution (deg.)', x='Orbiter distance (R$_P$)', 
             hue='Wavelength (cm)', ax=ax, palette=sns.color_palette('muted'))
ax.set_yscale('log')
ax.set_title('Aperture diameter: 3.7 m')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.get_legend().set_title('Wavelength (cm)')

fig.tight_layout()
fig.savefig('hga_latres.png', dpi=300)



# # Moon resolution 
# frequency = np.linspace(300e6, 30e9, 1000)
# wave = spc.c / frequency
# dist = [50, 85, 100]  # km
# resolution = [150]  # km 

# ls = ['-', '--']
# c = plt.cm.magma(np.linspace(0, 0.8, len(dist)))
# plt.figure(figsize=(14, 7))
# for i, d in enumerate(dist): 
#     for j, r in enumerate(resolution): 
#         res_dist = r  # km 
#         inst_angle = np.degrees(2 * np.arctan(res_dist / 2 / d))
#         inst_diam = 1. * wave / (np.radians(inst_angle))  # m, note this approximation is off, see Ulaby/Long discussion on antennas 
#         plt.loglog(wave, inst_diam, ls=ls[j], color=c[i], label='Dist.: {:.0f} km'.format(d))
#         print('Distance: {}, Spatial resolution: {}, Instrument beam angle: {}'.format(d, res_dist, inst_angle))

# plt.gca().invert_xaxis()
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.xlabel('Wavelength (meters)')
# plt.ylabel('Antenna diameter (meters)')

# ylim = plt.gca().get_ylim()
# xlim = plt.gca().get_xlim()
# juno_wave = np.array([0.5, 0.24, 0.1155, 0.0575, 0.03, 0.0137])
# juno_diams = np.array([1.6, 0.77, 0.72, 0.36, 0.18, 0.15])
# # plt.vlines(juno_wave, *ylim, color='k')
# plt.hlines(3.7, *xlim, color='r', ls='-')
# plt.scatter(juno_wave, juno_diams, c='k', s=20)
# # plt.gca().set_ylim(*ylim)
# plt.gca().set_xlim(*xlim)
# plt.tight_layout()
# plt.savefig('juno_moon_comparison.png', dpi=300)


