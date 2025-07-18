import numpy as np 
import scipy.constants as spc
import scipy.integrate as sint 

import molecule as mlc 


# 888b     d8888888888b.88888888888888b     d888 
# 8888b   d8888888   Y88b   888    8888b   d8888 
# 88888b.d88888888    888   888    88888b.d88888 
# 888Y88888P888888   d88P   888    888Y88888P888 
# 888 Y888P 8888888888P"    888    888 Y888P 888 
# 888  Y8P  888888 T88b     888    888  Y8P  888 
# 888   "   888888  T88b    888    888   "   888 
# 888       888888   T88b   888    888       888 

#  MRTM - A Giant Planet Microwave Radiative Transfer Model (non-scattering)


def mrtm(freq_ghz, incang_deg, z_km, p_pa, T, x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, x_h2, x_he, planet_radius): 
    """ Microwave radiative transfer model (non-scattering)

        Inputs: 
        freq_ghz: Frequencies in GHz, arbitrary number 
        incang_deg: Incidence angle in degrees, only one
        z_km: Altitude grid in km 
        p_pa: Atmospheric pressure grid in Pascals
        T: Atmospheric temperature grid in Kelvins 
        x_h2o: Vertical profile of water vapor, mole fraction
        x_nh3: Vertical profile of ammonia vapor, mole fraction
        x_h2s: Vertical profile of hydrogen sulfide vapor, mole fraction
        x_ch4: Vertical profile of methane vapor, mole fraction
        x_ph3: Vertical profile of phosphine vapor, mole fraction
        x_h2: Vertical profile of hydrogen, mole fraction
        x_he: Vertical profile of helium, mole fraction
        planet_radius: Radius of the planet in km (used for refractive bending calculation)

        Returns: 
        full_tau - Integrated optical depth of the atmosphere, 1/km
        weight_up - Atmospheric weighting functions (nadir/low-angle sounding)
        weight_limb - Atmospheric weighting functions (limb/high-angle sounding)
        Tup - Brightness temperature (nadir/low-angle sounding)
        Tlimb - Brightness temperature (limb/high-angle sounding)

    """
    # Sort so that first index is shallowest pressure 
    sortmask = np.argsort(p_pa)
    p_pa = p_pa[sortmask]  
    z_km = z_km[sortmask]  
    T = T[sortmask]  
    x_h2o = x_h2o[sortmask]  
    x_nh3 = x_nh3[sortmask]  
    x_h2s = x_h2s[sortmask]  
    x_ch4 = x_ch4[sortmask]  
    x_ph3 = x_ph3[sortmask]  
    x_h2 = x_h2[sortmask]  
    x_he = x_he[sortmask]  

    p_bar = p_pa / spc.bar 

    H2O = mlc.H2O()
    NH3 = mlc.NH3()
    H2S = mlc.H2S()
    CH4 = mlc.CH4()
    PH3 = mlc.PH3()
    H2 = mlc.H2() 
    He = mlc.He() 

    gases = np.vstack([x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, x_h2, x_he])
    gases_index = ['H2O', 'NH3', 'H2S', 'CH4', 'PH3', 'H2', 'He']
    
    # Computing absorption can be expensive. Skip it if not needed 
    zero_ref = np.zeros(z_km.shape)
    if np.isclose(x_h2o, zero_ref).all(): 
        h2o_alpha = np.zeros((len(freq_ghz), 1)) * zero_ref
    else: 
        h2o_alpha = H2O.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')

    if np.isclose(x_nh3, zero_ref).all():     
        nh3_alpha = np.zeros((len(freq_ghz), 1)) * zero_ref
    else: 
        nh3_alpha = NH3.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')
    
    if np.isclose(x_h2s, zero_ref).all():     
        h2s_alpha = np.zeros((len(freq_ghz), 1)) * zero_ref
    else: 
        h2s_alpha = H2S.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')
    
    if np.isclose(x_ch4, zero_ref).all():     
        ch4_alpha = np.zeros((len(freq_ghz), 1)) * zero_ref
    else: 
        ch4_alpha = CH4.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')
    
    if np.isclose(x_ph3, zero_ref).all():     
        ph3_alpha = np.zeros((len(freq_ghz), 1)) * zero_ref
    else: 
        ph3_alpha = PH3.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')

    # Bulk gases should never be zero 
    h2_alpha = H2.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')
    he_alpha = He.absorption(freq_ghz * 1e3, gases, gases_index, T, p_bar, units='invcm')

    alpha = np.sum(np.stack([h2o_alpha, nh3_alpha, h2s_alpha, ch4_alpha, ph3_alpha, h2_alpha, he_alpha]), axis=0)

    # Others don't have refractivity defined 
    h2o_n = H2O.refractivity(T, x_h2o * p_bar)
    ch4_n = CH4.refractivity(T, x_h2o * p_bar)
    h2_n = H2.refractivity(T, x_h2 * p_bar)
    he_n = He.refractivity(T, x_he * p_bar)
    nr = 1 + 1e-6 * np.sum(np.stack([h2o_n, ch4_n, h2_n, he_n]), axis=0)
    
    # Refractive path bending
    altr = planet_radius - z_km
    k = np.max(altr) * np.sin(np.radians(incang_deg))
    total_path = sint.cumulative_trapezoid((nr * altr) / np.sqrt((nr * altr)**2 - k**2), x=altr, initial=0)
    pathlength = np.gradient(total_path)
    pathlength[np.isnan(pathlength)] = np.inf

    # Main RT integrations
    dz = pathlength
    mask = np.isfinite(dz)
    z = sint.cumulative_trapezoid(-dz[mask], initial=0.)  # self.ray.altitude
    valid_alpha = alpha[..., mask] * 1e5 # 1e5 converts from cm-1 to km-1 
    up_tau = sint.cumulative_trapezoid(valid_alpha, x=z, axis=-1, initial=0)  
    down_tau = -sint.cumulative_trapezoid(valid_alpha[..., ::-1], x=z[::-1], axis=-1, initial=0)[..., ::-1]  
    full_tau = up_tau[..., -1]
    wup = valid_alpha * np.exp(-up_tau)
    wlimb = wup * (1 + np.exp(-2 * down_tau))
    Tup = sint.trapezoid(wup * T[mask], x=z, axis=-1)
    Tlimb = 2.73 * np.exp(-2 * full_tau) + sint.trapezoid(wlimb * T[mask], x=z, axis=-1)
    weight_up = np.zeros((len(freq_ghz), len(T)))
    weight_limb = np.zeros((len(freq_ghz), len(T)))
    weight_up[..., mask] = wup
    weight_limb[..., mask] = wlimb
    
    # Restore input convention for weighting functions 
    weight_up = weight_up[:, np.argsort(sortmask)]
    weight_limb = weight_limb[:, np.argsort(sortmask)]
    return full_tau, weight_up, weight_limb, Tup, Tlimb
