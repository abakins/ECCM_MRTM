import numpy as np 
import scipy.constants as spc 
from numba import njit 

import thermo_data as thermo
import find_root


@njit
def run_eccm(pressure_grid, temperature_grid, 
             deep_h2o, deep_nh3, deep_h2s, deep_ch4, deep_ph3,
             h2o_rh, nh3_rh, h2s_rh, ch4_rh, ph3_rh,
             bulk_h2, bulk_he, latent_heat_update, force_temperature=False): 
    
    x_h2o = np.zeros(pressure_grid.shape)
    x_nh3 = np.zeros(pressure_grid.shape)
    x_h2s = np.zeros(pressure_grid.shape)
    x_ch4 = np.zeros(pressure_grid.shape)
    x_ph3 = np.zeros(pressure_grid.shape)
    
    a_h2osolid = np.zeros(pressure_grid.shape)
    a_h2oliquid = np.zeros(pressure_grid.shape)
    a_h2osolution = np.zeros(pressure_grid.shape)
    a_nh3solid = np.zeros(pressure_grid.shape)
    a_nh3liquid = np.zeros(pressure_grid.shape)
    a_h2ssolid = np.zeros(pressure_grid.shape)
    a_h2sliquid = np.zeros(pressure_grid.shape)
    a_nh4sh = np.zeros(pressure_grid.shape)
    c_nh3h2o = np.zeros(pressure_grid.shape)
    a_ch4solid = np.zeros(pressure_grid.shape)
    a_ch4liquid = np.zeros(pressure_grid.shape)
    a_ph3solid = np.zeros(pressure_grid.shape)
    
    x_h2o[0] = deep_h2o 
    x_nh3[0] = deep_nh3
    x_h2s[0] = deep_h2s
    x_ch4[0] = deep_ch4
    x_ph3[0] = deep_ph3

    if (deep_nh3 > 0) and (deep_h2o > 0): 
        # First guess at concentration
        c_nh3h2o[0] = deep_nh3 / (deep_nh3 + deep_h2o)

    for i in range(len(pressure_grid) - 1):
        p1 = pressure_grid[i]
        p2 = pressure_grid[i+1]
        pmean = 0.5 * (p1 + p2)
        dp = p2 - p1
        T1 = temperature_grid[i]
        x1_h2o = x_h2o[i]
        x1_nh3 = x_nh3[i]
        x1_h2s = x_h2s[i]
        x1_ch4 = x_ch4[i]
        x1_ph3 = x_ph3[i]
        c1_nh3h2o = c_nh3h2o[i]

        if force_temperature: 
            T2 = temperature_grid[i + 1]
        else: 
            # Update temperature, dry adiabatic lapse rate 
            cp = bulk_h2 * thermo.h2_normal_molar_heat_capacity(T1) + bulk_he * thermo.he_molar_heat_capacity
            dTdp = spc.R * T1 / cp / pmean
            T2 = T1 + dTdp * dp 

        # Check all gases 
        x2_h2o, x2_nh3, x2_h2s, x2_ch4, x2_ph3, \
        a2_h2osolid, a2_h2oliquid, a2_h2osolution, \
        a2_nh3solid, a2_nh3liquid, \
        a2_h2ssolid, a2_h2sliquid, \
        a2_nh4sh, c2_nh3h2o, \
        a2_ch4solid, a2_ch4liquid, a2_ph3solid, \
        lh_h2o, lh_nh3, lh_h2s, lh_ch4, lh_ph3, lh_nh4sh = gas_cloud_T(p1, T1, p2, T2, x1_h2o, x1_nh3, x1_h2s, x1_ch4, x1_ph3, c1_nh3h2o)


        # Recalculate temperature, wet adiabatic lapse rate 
        # If condensation didn't occur for a particular gas, 
        # returned latent heats will be set to zero and won't impact 
        # the update
        if latent_heat_update and not force_temperature: 
            dTdp_numerator = spc.R * T1 + lh_h2o * x1_h2o + lh_nh3 * x1_nh3 + lh_h2s * x1_h2s + lh_ch4 * x1_ch4 + lh_ph3 * x1_ph3 
            dTdp_denominator = pmean * cp + pmean / spc.R / T1**2 * \
                               (lh_h2o**2 * x1_h2o + lh_nh3**2 * x1_nh3 + lh_h2s**2 * x1_h2s + lh_ch4**2 * x1_ch4 + lh_ph3**2 * x1_ph3)
            if (x1_nh3 > 0) and (x1_h2s > 0): 
                dTdp_numerator = dTdp_numerator + 2 * lh_nh4sh * x1_nh3 * x1_h2s / (x1_nh3 + x1_h2s) 
                dTdp_denominator = dTdp_denominator + pmean * lh_nh4sh * x1_nh3 * x1_h2s / (x1_nh3 + x1_h2s) * 10834 / T1**2 
        
            dTdp = dTdp_numerator / dTdp_denominator
            T2 = T1 + dTdp * dp 

            # Check all gases again
            x2_h2o, x2_nh3, x2_h2s, x2_ch4, x2_ph3, \
            a2_h2osolid, a2_h2oliquid, a2_h2osolution, \
            a2_nh3solid, a2_nh3liquid, \
            a2_h2ssolid, a2_h2sliquid, \
            a2_nh4sh, c2_nh3h2o, \
            a2_ch4solid, a2_ch4liquid, a2_ph3solid, \
            lh_h2o, lh_nh3, lh_h2s, lh_ch4, lh_ph3, lh_nh4sh = gas_cloud_T(p1, T1, p2, T2, x1_h2o, x1_nh3, x1_h2s, x1_ch4, x1_ph3, c1_nh3h2o)

        # Set it, and press onwards
        temperature_grid[i + 1] = T2 
        x_h2o[i + 1] = x2_h2o
        x_nh3[i + 1] = x2_nh3
        x_h2s[i + 1] = x2_h2s
        x_ch4[i + 1] = x2_ch4
        x_ph3[i + 1] = x2_ph3
        a_h2osolid[i + 1] = a2_h2osolid
        a_h2oliquid[i + 1] = a2_h2oliquid
        a_h2osolution[i + 1] = a2_h2osolution
        a_nh3solid[i + 1] = a2_nh3solid
        a_nh3liquid[i + 1] = a2_nh3liquid
        a_h2ssolid[i + 1] = a2_h2ssolid
        a_h2sliquid[i + 1] = a2_h2sliquid
        a_nh4sh[i + 1] = a2_nh4sh
        c_nh3h2o[i + 1] = c2_nh3h2o
        a_ch4solid[i + 1] = a2_ch4solid
        a_ch4liquid[i + 1] = a2_ch4liquid
        a_ph3solid[i + 1] = a2_ph3solid

    # Finally, update the relative humidities 
    mask = (a_h2osolid + a_h2oliquid + a_h2osolution).astype(np.bool)
    x_h2o[mask] = x_h2o[mask] * h2o_rh
    mask = (a_nh3solid + a_nh3liquid).astype(np.bool)
    x_nh3[mask] = x_nh3[mask] * nh3_rh
    mask = (a_h2ssolid + a_h2sliquid).astype(np.bool)
    x_h2s[mask] = x_h2s[mask] * h2s_rh
    mask = (a_ch4solid + a_ch4liquid).astype(np.bool)
    x_ch4[mask] = x_ch4[mask] * ch4_rh
    mask = (a_ph3solid).astype(np.bool)
    x_ph3[mask] = x_ph3[mask] * ph3_rh

    return temperature_grid, x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, \
            a_h2osolid, a_h2oliquid, a_h2osolution, \
            a_nh3solid, a_nh3liquid, \
            a_h2ssolid, a_h2sliquid, \
            a_nh4sh, c_nh3h2o, a_ch4solid, a_ch4liquid, a_ph3solid
    

@njit
def gas_cloud_T(p1, T1, p2, T2, x1_h2o, x1_nh3, x1_h2s, x1_ch4, x1_ph3, c1_nh3h2o): 
    # Where all the fun happens 
    # Note to user - This was not written to be sophisticated, it was written to work

    threshold = 1e-12  # If gas mole fractions fall below this value, they are forced to zero
    nh4sh_condensation_flag = False 

    # Simple ones first 
    if x1_ch4 > 0: 
        # CH4 condensation
        x2_ch4, lh_ch4, solid_flag = update_ch4(p1, T1, p2, T2, x1_ch4)
        uc = update_cloud(p1, p2, T1, x1_ch4, x2_ch4, thermo.ch4_molar_mass)
        if solid_flag: 
            a2_ch4solid = uc 
            a2_ch4liquid = 0.
        else: 
            a2_ch4solid = 0. 
            a2_ch4liquid = uc
    else: 
        x2_ch4 = 0.
        lh_ch4 = 0. 
        a2_ch4solid = 0. 
        a2_ch4liquid = 0. 

    if x1_ph3 > 0.: 
        # PH3 condensation
        x2_ph3, lh_ph3 = update_ph3(p1, T1, p2, T2, x1_ph3)
        a2_ph3solid = update_cloud(p1, p2, T1, x1_ph3, x2_ph3, thermo.ph3_molar_mass)
    else: 
        x2_ph3 = 0.
        lh_ph3 = 0. 
        a2_ph3solid = 0. 

    # The rest...

    if x1_h2o == 0.:  
        # No water
        lh_h2o = 0. 
        x2_h2o = 0.          
        a2_h2osolid = 0. 
        a2_h2oliquid = 0. 
        a2_h2osolution = 0.
        c2_nh3h2o = 0.
        
        if x1_nh3 == 0.: 
            # No ammonia either?
            x2_nh3 = 0.
            lh_nh3 = 0. 
            a2_nh3solid = 0.
            a2_nh3liquid = 0.
            lh_nh4sh = 0.
            a2_nh4sh = 0.

            if x1_h2s == 0.: 
                # No hydrogen sulfide either???
                x2_h2s = 0.
                lh_h2s = 0. 
                a2_h2ssolid = 0.
                a2_h2sliquid = 0.
            else: 
                # Simple H2S condensation
                x2_h2s, lh_h2s, solid_flag = update_h2s(p1, T1, p2, T2, x1_h2s)
                uc = update_cloud(p1, p2, T1, x1_h2s, x2_h2s, thermo.h2s_molar_mass)
                if solid_flag: 
                    a2_h2ssolid = uc 
                    a2_h2sliquid = 0.
                else: 
                    a2_h2ssolid = 0. 
                    a2_h2sliquid = uc
        else: 
            # There is ammonia
            if x1_h2s == 0.: 
                # Simple NH3 condensation
                x2_h2s = 0.
                lh_h2s = 0. 
                a2_h2ssolid = 0.
                a2_h2sliquid = 0.
                lh_nh4sh = 0.
                a2_nh4sh = 0.
                x2_nh3, lh_nh3, solid_flag = update_nh3(p1, T1, p2, T2, x1_nh3)
                uc = update_cloud(p1, p2, T1, x1_nh3, x2_nh3, thermo.nh3_molar_mass)
                if solid_flag: 
                    a2_nh3solid = uc 
                    a2_nh3liquid = 0.
                else: 
                    a2_nh3solid = 0. 
                    a2_nh3liquid = uc
            else:
                # NH4SH condensation
                lh_nh3 = 0.
                lh_h2s = 0.
                a2_nh3solid = 0.
                a2_nh3liquid = 0.
                a2_h2ssolid = 0.
                a2_h2sliquid = 0.
                
                x2_h2s, x2_nh3, lh_nh4sh = update_nh4sh(p1, T1, p2, T2, x1_h2s, x1_nh3)
                a2_nh4sh = update_cloud(p1, p2, T1, x1_h2s, x2_h2s, thermo.nh4sh_molar_mass)
               
    else: 
        # There is water 
        if x1_nh3 == 0.: 
            # Simple H2O condensation
            x2_nh3 = 0 
            lh_nh3 = 0 
            a2_nh3solid = 0 
            a2_nh3liquid = 0 
            a2_nh4sh = 0. 
            c2_nh3h2o = 0. 
            x2_h2o, lh_h2o, solid_flag = update_h2o(p1, T1, p2, T2, x1_h2o)
            uc = update_cloud(p1, p2, T1, x1_h2o, x2_h2o, thermo.h2o_molar_mass)
            if solid_flag: 
                a2_h2osolid = uc 
                a2_h2oliquid = 0.
            else: 
                a2_h2osolid = 0. 
                a2_h2oliquid = uc
            a2_h2osolution = 0.
           
            if x1_h2s == 0.: 
                x2_h2s = 0.
                lh_h2s = 0. 
                a2_h2ssolid = 0.
                a2_h2sliquid = 0.
            else: 
                # Simple H2S condensation
                x2_h2s, lh_h2s, solid_flag = update_h2s(p1, T1, p2, T2, x1_h2s)
                uc = update_cloud(p1, p2, T1, x1_h2s, x2_h2s, thermo.h2s_molar_mass)
            if solid_flag: 
                a2_h2ssolid = uc 
                a2_h2sliquid = 0.
            else: 
                a2_h2ssolid = 0. 
                a2_h2sliquid = uc

        else: 
            # H2O - NH3 solution condensation
            # Note the use of an intermediate mole fraction in this block (x12) in case different condensation steps happen
            # Practically they won't, but I did this for some reason anyway 
            x2_h2o, x12_nh3, lh_h2o, lh_nh3, c2_nh3h2o, solid_nh3_flag, solid_h2o_flag, solution_flag = update_nh3_h2o_solution(p1, T1, p2, T2, x1_h2o, x1_nh3, c1_nh3h2o)
            uc1 = update_cloud(p1, p2, T1, x1_h2o, x2_h2o, thermo.h2o_molar_mass)
            uc2 = update_cloud(p1, p2, T1, x1_nh3, x12_nh3, thermo.nh3_molar_mass)
            
            if solution_flag: 
                a2_h2osolution = uc1 + uc2
                a2_h2osolid = 0. 
                a2_nh3solid = 0. 
                a2_h2oliquid = 0. 
                a2_nh3liquid = 0. 
                 
            else: 
                a2_h2osolution = 0. 
                c2_nh3h2o = c1_nh3h2o
                if solid_h2o_flag:
                    a2_h2osolid = uc1
                    a2_h2oliquid = 0. 
                else: 
                    a2_h2osolid = 0.
                    a2_h2oliquid = uc1

                if solid_nh3_flag:
                    a2_nh3solid = uc2
                    a2_nh3liquid = 0. 
                else: 
                    a2_nh3solid = 0.
                    a2_nh3liquid = uc2

            if x1_h2s > 0: 
                # H2S dissolution into condensed H2O - NH3 
                if solution_flag: 
                    dxh2o = x2_h2o - x1_h2o
                    dxnh3 = x12_nh3 - x1_nh3
                    dxh2s, ch2s = h2s_nh3h2osolution_difference(T2, p2, dxh2o, dxnh3, x1_h2s)
                else: 
                    dxh2s = 0 

                x12_h2s = x1_h2s + dxh2s 
                
                # NH4SH condensation, after any water dissolution has happened
                x2_h2s, x2_nh3, lh_nh4sh = update_nh4sh(p1, T1, p2, T2, x12_h2s, x12_nh3)
                a2_nh4sh = update_cloud(p1, p2, T1, x12_h2s, x2_h2s, thermo.nh4sh_molar_mass)
                    
            else: 
                x2_nh3 = x12_nh3
                x2_h2s = 0. 
                lh_h2s = 0.     
                lh_nh4sh=0.
                a2_nh4sh = 0. 
            a2_h2ssolid = 0.
            a2_h2sliquid = 0.

    # Truncate
    if x2_ph3 < threshold: 
        x2_ph3 = 0. 
        lh_ph3 = 0. 
        a2_ph3solid = 0. 
    if x2_ch4 < threshold: 
        x2_ch4 = 0. 
        lh_ch4 = 0. 
        a2_ch4solid = 0. 
        a2_ch4liquid = 0. 
    if x2_h2o < threshold: 
        x2_h2o = 0. 
        lh_h2o = 0. 
        a2_h2osolid = 0. 
        a2_h2oliquid = 0. 
        a2_h2osolution = 0. 
    if x2_nh3 < threshold: 
        x2_nh3 = 0. 
        lh_nh3 = 0. 
        a2_nh3solid = 0. 
        a2_nh3liquid = 0. 
        lh_nh4sh = 0. 
        a2_nh4sh = 0. 
    if x2_h2s < threshold: 
        x2_h2s = 0. 
        lh_h2s = 0. 
        a2_h2ssolid = 0. 
        a2_h2sliquid = 0. 
        lh_nh4sh = 0. 
        a2_nh4sh = 0. 

    return x2_h2o, x2_nh3, x2_h2s, x2_ch4, x2_ph3,  \
           a2_h2osolid, a2_h2oliquid, a2_h2osolution, \
           a2_nh3solid, a2_nh3liquid, \
           a2_h2ssolid, a2_h2sliquid, \
           a2_nh4sh, c2_nh3h2o, \
           a2_ch4solid, a2_ch4liquid, a2_ph3solid, \
           lh_h2o, lh_nh3, lh_h2s, lh_ch4, lh_ph3, lh_nh4sh 


# Evaluation functions for condensation occurrence 
# Gas update conventions are referenced to "level 1" - deeper pressure, and "level 2" - shallower pressure 

# "Vanilla" updates 
@njit
def update_ph3(p1, T1, p2, T2, x1):
    dT = T2-T1
    dp = p2-p1
    # Solid only 
    svp, lh = thermo.ph3_solid_saturation_vapor_pressure(T2)
    svp = svp * 1e5  # bar to Pa
    if (svp / p2) < x1: 
        dx = x1 * lh / spc.R / T2**2 * dT - x1 / p2 * dp
        dx = max(dx, (svp / p2) - x1)
    else: 
        lh = 0.
        dx = 0. 
    x2 = x1 + dx 
    return x2, lh 

@njit
def update_ch4(p1, T1, p2, T2, x1):
    dT = T2-T1
    dp = p2-p1
    if T2 < thermo.ch4_triple_point: 
        solid_flag = True
        svp, lh = thermo.ch4_solid_saturation_vapor_pressure(T2)
    else: 
        solid_flag = False
        svp, lh = thermo.ch4_liquid_saturation_vapor_pressure(T2)
    svp = svp * 1e5  # Bar to Pa
    if (svp / p2) < x1: 
        dx = x1 * lh / spc.R / T2**2 * dT - x1 / p2 * dp
        dx = max(dx, (svp / p2) - x1)
    else: 
        dx = 0.
        lh = 0.
    x2 = x1 + dx 
    return x2, lh, solid_flag

@njit
def update_h2s(p1, T1, p2, T2, x1):
    dT = T2-T1
    dp = p2-p1
    if T2 < thermo.h2s_triple_point: 
        solid_flag = True
        svp, lh = thermo.h2s_solid_saturation_vapor_pressure(T2)
    else: 
        solid_flag = False
        svp, lh = thermo.h2s_liquid_saturation_vapor_pressure(T2)
    svp = svp * 1e5  # Bar to Pa
    if (svp / p2) < x1: 
        dx = x1 * lh / spc.R / T2**2 * dT - x1 / p2 * dp
        dx = max(dx, (svp / p2) - x1)
    else: 
        dx = 0.
        lh = 0.
    x2 = x1 + dx 
    return x2, lh, solid_flag 

@njit
def update_nh3(p1, T1, p2, T2, x1):
    dT = T2-T1
    dp = p2-p1
    if T2 < thermo.nh3_triple_point: 
        solid_flag = True
        svp, lh = thermo.nh3_solid_saturation_vapor_pressure(T2)
    else: 
        solid_flag = False
        svp, lh = thermo.nh3_liquid_saturation_vapor_pressure(T2)
    svp = svp * 1e5  # Bar to Pa
    if (svp / p2) < x1: 
        dx = x1 * lh / spc.R / T2**2 * dT - x1 / p2 * dp
        dx = max(dx, (svp / p2) - x1)
    else: 
        dx = 0
    x2 = x1 + dx 
    return x2, lh, solid_flag

@njit
def update_h2o(p1, T1, p2, T2, x1):
    dT = T2-T1
    dp = p2-p1
    if T2 < thermo.h2o_triple_point: 
        solid_flag = True
        svp, lh = thermo.h2o_solid_saturation_vapor_pressure(T2)
    else: 
        solid_flag = False
        svp, lh = thermo.h2o_liquid_saturation_vapor_pressure(T2)
    svp = svp * 1e5  # Bar to Pa
    if (svp / p2) < x1: 
        dx = x1 * lh / spc.R / T2**2 * dT - x1 / p2 * dp
        dx = max(dx, (svp / p2) - x1)
    else: 
        dx = 0. 
        lh = 0.
    x2 = x1 + dx 
    return x2, lh, solid_flag

@njit
def update_cloud(p1, p2, T2, x1, x2, molar_mass):
    dx = x2 - x1 
    dp = p2 - p1 
    aer = molar_mass * dx * p2**2 / (spc.R * T2 * dp)  # Units are g/m3 
    return aer

# NH4SH
@njit
def update_nh4sh(p1, T1, p2, T2, x1_h2s, x1_nh3): 
    dT = T2-T1
    dp = p2-p1
    nh3p = p2 / spc.bar * x1_nh3
    h2sp = p2 / spc.bar * x1_h2s
    eqr = 34.150 - 10834 / T2
    if np.log(nh3p * h2sp) > eqr: 
        dx = (x1_h2s * x1_nh3) / (x1_nh3 + x1_h2s) * (10834 * dT / T2**2 - 2 * dp / p2)
        lh = thermo.nh4sh_latent_heat
        s1, s2 = find_root.solve_quadratic(1., x1_h2s + x1_nh3, x1_h2s * x1_nh3 - np.exp(eqr) / (p2 / spc.bar)**2)
        dxr = max(s1, s2)
        dx = max(dx, dxr)
        ### Derivation 
        # dX can't bring mole fractions lower than the saturation value, otherwise there will be cloud skipping, also done in other functions above
        # The maximum dX satisfies log(p2(xn + dx)p2(xh+dx)) = eqr 
        # This sets up a quadratic 
        # dx^2 + (xn + xh)dx + xn*xh = exp(eqr - 2logp) = exp(eqr) / p^2 
        # Solving this quadratic gives dxr, which is the dx which meets the saturation point. 
        ### 

    else: 
        dx = 0.
        lh = 0.
        x2_h2s = x1_h2s
        x2_nh3 = x1_nh3 

    x2_h2s = x1_h2s + dx
    x2_nh3 = x1_nh3 + dx

    return x2_h2s, x2_nh3, lh

@njit
def update_nh3_h2o_solution(p1, T1, p2, T2, x1_h2o, x1_nh3, c1_nh3h2o): 

    dT = T2-T1
    dp = p2-p1
    
    # Check the vapor pressures, does solution condensation occur? 
    Tf = thermo.h2o_nh3h2osolution_freezing_point(c1_nh3h2o) 
    if T2 < Tf:
        # Water is freezing, solution condensation does not occur
        x2_h2o, lh_h2o, solid_h2o_flag = update_h2o(p1, T1, p2, T2, x1_h2o)    
        x2_nh3, lh_nh3, solid_nh3_flag = update_nh3(p1, T1, p2, T2, x1_nh3)    
        solution_flag = False
        c2_nh3h2o = c1_nh3h2o
    else: 
        # Solution condensation might occur 
        pnh3 = x1_nh3 * p2
        ph2o = x1_h2o * p2

        # Check bracketing concentrations for pure constituent 
        svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T2, concentration=1.)[0] * 1e5 
        if pnh3 >= svp_nh3: 
            c_max = 1. 
        else: 
            c_max = find_root.brent(fn_nh3, 0., 1., args=(pnh3, T2))[0]

        svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T2, concentration=0.)[0] * 1e5 
        if ph2o >= svp_h2o: 
            c_min = 0.
        else: 
            c_min = find_root.brent(fn_h2o, 0., 1., args=(ph2o, T2))[0]
        if c_max < c_min: 
            c_min, c_max = c_max, c_min 

        # c_min can't be exactly zero or one, or else bracketing breaks 
        c = find_root.brent(fn_both, c_min+1e-6, c_max-1e-6, args=(pnh3, ph2o, T2))[0]
        eval_c = fn_both(c, pnh3, ph2o, T2)
        eval_allh2o = fn_both(0, pnh3, ph2o, T2)
        eval_allnh3 = fn_both(1, pnh3, ph2o, T2)

        if (eval_allh2o < eval_allnh3) and (eval_allh2o < eval_c): 
            conc = 0. 
        elif (eval_allnh3 < eval_allh2o) and (eval_allnh3 < eval_c): 
            conc = 1. 
        else: 
            conc = c  
        
        if np.isclose(conc, 1.): 
            # No good solution, check pure ammonia 
            x2_nh3, lh_nh3, solid_nh3_flag = update_nh3(p1, T1, p2, T2, x1_nh3)    
            solution_flag = False
            c2_nh3h2o = c1_nh3h2o
            x2_h2o = x1_h2o 
            lh_h2o = 0
        elif np.isclose(conc, 0.): 
            # No good solution, check pure water
            x2_h2o, lh_h2o, solid_h2o_flag = update_h2o(p1, T1, p2, T2, x1_h2o) 
            solution_flag = False
            c2_nh3h2o = c1_nh3h2o
            x2_nh3 = x1_nh3 
            lh_nh3 = 0
        else: 
            # A concentration was found 
            svp_h2o, lh_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T2, concentration=conc)
            svp_nh3, lh_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T2, concentration=conc)
            svp_h2o = svp_h2o * 1e5 # Convert to bar
            svp_nh3 = svp_nh3 * 1e5
            if (svp_h2o / p2) < x1_h2o: 
                # Condensation is occuring, use current concentration to compute change in mole fraction
                dxw = x1_h2o * lh_h2o / spc.R / T2**2 * dT - x1_h2o / p2 * dp 
                dxw = max(dxw, (svp_h2o / p2) - x1_h2o)
                dxa = conc / (1 - conc) * dxw     
                c2_nh3h2o = conc
                solution_flag = True
            else: 
                # Condensation isn't occuring after all
                dxw = 0.
                dxa = 0.
                lh_h2o = 0. 
                lh_nh3 = 0. 
                c2_nh3h2o = c1_nh3h2o
                solution_flag = False

            x2_h2o = x1_h2o + dxw
            x2_nh3 = x1_nh3 + dxa
            solid_h2o_flag = False 
            solid_nh3_flag = False 
            
    return x2_h2o, x2_nh3, lh_h2o, lh_nh3, c2_nh3h2o, solid_nh3_flag, solid_h2o_flag, solution_flag

            
@njit 
def h2s_nh3h2osolution_difference(temperature, pressure, dxh2o, dxnh3, xh2s):
    """ Computes how much H2S dissolves into an ammonia solution cloud 
        Modified from function written by Paul Romani 
        :param temperature: Layer temperature in K
        :param pressure: Layer pressure in Pascal
        :param dxh2o: Dissolved water mole fraction into solution cloud 
        :param dxnh3: Dissolved water mole fraction into solution cloud 
        :param xh2s: Layer H2S volume mole fraction before dissolution 
        :return dxh2s: Change in H2S mole fraction for this layer  
        :return ch2s: Resulting H2S concentration in mol/L 
    """

    pressure = pressure / spc.mmHg  # Convert to mmHg 

    # Constants
    litre_of_cm3 = 1e-3

    # Find the density of the solution cloud
    X = (dxnh3 * thermo.nh3_molar_mass) / (dxnh3 * thermo.nh3_molar_mass + dxh2o * thermo.h2o_molar_mass)  # Mass fraction in g/g
    DENSOL = 0.9991 + X * (-0.4336 + X * (0.3303 + X * (0.2833 + X * (-1.9716 + X * (2.1396 - X * 0.7294)))))  # Solution density in g/cm3

    # Calculate the partial pressure of H2S in the atmosphere in mmHg
    PPH2S = xh2s * pressure

    # Find the concentration of the solution cloud in moles of NH3 / liter
    VOLSOL = -(dxh2o * thermo.h2o_molar_mass + dxnh3 * thermo.nh3_molar_mass) / DENSOL * litre_of_cm3  
    CONSOL = -dxnh3 / VOLSOL

    # Iterative method to calculate the concentration of H2S in solution
    FUNTMP = np.exp(22.221 - 5388.0 / temperature)
    FUNNH3 = CONSOL ** 1.8953
    F = FUNNH3 / FUNTMP

    POWER = 1.0 / (1.130 + 1.8953)
    OLDC = 0.0
    CH2S = 0.0

    for i in range(50):
        CH2S = (PPH2S * F) ** POWER
        ECH2S = abs((CH2S - OLDC) / CH2S) * 100.0
        if ECH2S <= 0.001:
            DFH2S = -(CH2S * VOLSOL)
            return DFH2S, CH2S
        OLDC = CH2S
        PPH2S = (xh2s - (CH2S * VOLSOL)) * pressure 
        if PPH2S <= 0.0:
            DFH2S = 0.0
            CH2S = 0.0
            return DFH2S, CH2S
    # If no convergence, return zero values
    return 0.0, 0.0

# Utilities for NH3/H2O solution concentration finding 
@njit 
def fn_nh3(gc, pnh3, T1): 
    svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T1, concentration=gc)[0] * 1e5 
    return abs(svp_nh3 - pnh3)

@njit 
def fn_h2o(gc, ph2o, T1): 
    svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T1, concentration=gc)[0] * 1e5 
    return abs(svp_h2o - ph2o)

@njit 
def fn_both(gc, pnh3, ph2o, T1): 
    if gc < 0: 
        gc = 0.
    elif gc > 1: 
        gc = 1. 
    svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T1, concentration=gc)[0] * 1e5 
    svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T1, concentration=gc)[0] * 1e5 
    return abs((1.0 - gc) * (svp_nh3 - pnh3) - gc * (svp_h2o - ph2o))