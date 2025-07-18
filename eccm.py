import sys 
import numpy as np 
import scipy.constants as spc 
import scipy.interpolate as spi 
import scipy.integrate as sint 

import thermo_data as thermo
from core import run_eccm


# 8888888888       ,o888888o.        ,o888888o.           ,8.       ,8.          
# 8888            8888     `88.     8888     `88.        ,888.     ,888.         
# 8888         ,8 8888       `8. ,8 8888       `8.      .`8888.   .`8888.        
# 8888         88 8888           88 8888               ,8.`8888. ,8.`8888.       
# 888888888888 88 8888           88 8888              ,8'8.`8888,8^8.`8888.      
# 8888         88 8888           88 8888             ,8' `8.`8888' `8.`8888.     
# 8888         88 8888           88 8888            ,8'   `8.`88'   `8.`8888.    
# 8888         `8 8888       .8' `8 8888       .8' ,8'     `8.`'     `8.`8888.   
# 8888            8888     ,88'     8888     ,88' ,8'       `8        `8.`8888.  
# 888888888888     `8888888P'        `8888888P'  ,8'         `         `8.`8888. 
   
#  ECCM - A Giant Planet Equilibrium Cloud Condensation Model


def eccm(pressure_grid, reference_pressure, reference_temperature, planet_gravity,
         deep_h2o, deep_nh3, deep_h2s, deep_ch4, deep_ph3, bulk_h2, bulk_he, 
         h2o_rh=1., nh3_rh=1., h2s_rh=1., ch4_rh=1., ph3_rh=1.,
         latent_heat_update=False, force_reference_above_pressure=1.*spc.bar): 
    
    """ Equilibrium cloud condensation model 
        Inputs: 
        pressure_grid: Pressure grid to work with, Pascals
        reference_pressure: Pressure points for reference_temperature, Pascals
        reference_temperature: Reference temperature profile, from occultations, Kelvins
        planet_gravity: Planet gravity, m/s^2, need for altitude calculation
        
        deep_h2o: Deep H2O mole fraction
        deep_nh3: Deep NH3 mole fraction 
        deep_h2s: Deep H2S mole fraction
        deep_ch4: Deep CH4 mole fraction
        deep_ph3: Deep PH3 mole fraction 
        bulk_h2: H2 bulk fraction 
        bulk_he: He bulk fraction 
        h2o_rh: Relative humidity fraction, H2O
        nh3_rh: Relative humidity fraction, NH3
        h2s_rh: Relative humidity fraction, H2S
        ch4_rh: Relative humidity fraction, CH4
        ph3_rh: Relative humidity fraction, PH3
        latent_heat_update: If True (default False), temperature profile updates will account for 
                            latent heat of condensation
        force_reference_above_pressure: Pressure in Pa above which temperature profile is forced to 
                                        match the reference profile, default is 1 bar level. 

        Note that returned aerosol densities (in g/m3) aren't particularly accurate 
        (They are accurate in the sense of original Weidenschilling/Lewis model, 
        but not accurate in general, see M. Wong et al. 2015 "fresh clouds" discussion)
        Neither is the NH3/H2O solution concentration below the cloud base 
        
        Gas profiles are generally accurate, which is important 
        for the main application of this code (computing microwave brightness temperatures)

    """

    pressure_grid = pressure_grid[np.argsort(pressure_grid)[::-1]]  # Sort so that first index is deepest pressure 
    temperature_grid = compute_temperature_guess(pressure_grid, reference_temperature, reference_pressure, bulk_h2, bulk_he)

    new_temperature_grid, x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, \
    a_h2osolid, a_h2oliquid, a_h2osolution, \
    a_nh3solid, a_nh3liquid, \
    a_h2ssolid, a_h2sliquid, \
    a_nh4sh, c_nh3h2o, a_ch4solid, a_ch4liquid, a_ph3solid = run_eccm(pressure_grid, temperature_grid, 
                                                                      deep_h2o, deep_nh3, deep_h2s, deep_ch4, deep_ph3,
                                                                      h2o_rh, nh3_rh, h2s_rh, ch4_rh, ph3_rh,
                                                                      bulk_h2, bulk_he, latent_heat_update) 

    # Loop to force temperature match to reference
    if force_reference_above_pressure is not None: 
        force_index = np.argmin(abs(pressure_grid - force_reference_above_pressure))
        ref_t_grid_func = spi.interp1d(reference_pressure, reference_temperature, kind='linear', bounds_error=False, fill_value='extrapolate')
        offs = new_temperature_grid[force_index] - ref_t_grid_func(pressure_grid[force_index])
        
        off_count = 0 
        while abs(offs) > 1e-2:
            off_count = off_count + 1 
            if off_count > 250: 
                print("""Temperature profile matching iteration is either diverging or oscillating, 
                         printing current offset and moving ahead anyway""")
                print(offs)
                break 
            new_temperature_grid[0] = new_temperature_grid[0] - offs
            new_temperature_grid, x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, \
            a_h2osolid, a_h2oliquid, a_h2osolution, \
            a_nh3solid, a_nh3liquid, \
            a_h2ssolid, a_h2sliquid, \
            a_nh4sh, c_nh3h2o, a_ch4solid, a_ch4liquid, a_ph3solid = run_eccm(pressure_grid, new_temperature_grid, 
                                                                            deep_h2o, deep_nh3, deep_h2s, deep_ch4, deep_ph3, 
                                                                            h2o_rh, nh3_rh, h2s_rh, ch4_rh, ph3_rh,
                                                                            bulk_h2, bulk_he, latent_heat_update) 
            offs = new_temperature_grid[force_index] - ref_t_grid_func(pressure_grid[force_index])
        new_temperature_grid[force_index:] = ref_t_grid_func(pressure_grid[force_index:])

        # And finally run once more, forcing the temperature profile
        new_temperature_grid, x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, \
        a_h2osolid, a_h2oliquid, a_h2osolution, \
        a_nh3solid, a_nh3liquid, \
        a_h2ssolid, a_h2sliquid, \
        a_nh4sh, c_nh3h2o, a_ch4solid, a_ch4liquid, a_ph3solid = run_eccm(pressure_grid, new_temperature_grid, 
                                                                          deep_h2o, deep_nh3, deep_h2s, deep_ch4, deep_ph3,
                                                                          h2o_rh, nh3_rh, h2s_rh, ch4_rh, ph3_rh,
                                                                          bulk_h2, bulk_he, latent_heat_update, force_temperature=True) 

    # Compute altitude/pressure mapping
    altitude_grid = hypsometric(pressure_grid, new_temperature_grid, planet_gravity, bulk_h2, bulk_he)

    # Adjust H2/He to maintain the appropriate split 
    total_x = x_h2o + x_nh3 + x_h2s + x_ch4 + x_ph3
    ratio_h2 = bulk_h2 / (1 - bulk_h2) 
    x_h2 = (1 - total_x) / (1 + 1 / ratio_h2)
    ratio_he = bulk_he / (1 - bulk_he) 
    x_he = (1 - total_x) / (1 + 1 / ratio_he)
    
    return pressure_grid, new_temperature_grid, altitude_grid, \
            x_h2o, x_nh3, x_h2s, x_ch4, x_ph3, x_h2, x_he, \
            a_h2osolid, a_h2oliquid, a_h2osolution, \
            a_nh3solid, a_nh3liquid, \
            a_h2ssolid, a_h2sliquid, \
            a_nh4sh, c_nh3h2o, a_ch4solid, a_ch4liquid, a_ph3solid 

def compute_temperature_guess(pressure_grid, reference_temperature, reference_pressure, bulk_h2, bulk_he): 
    """ Extrapolates a reference temperature profile assuming the dry adiabatic lapse rate

        pressure_grid: Pressure grid to work with, Pascals
        reference_temperature: Reference temperature profile, from occultations, Kelvins
        reference_pressure: Pressure points for reference_temperature, Pascals
        bulk_h2: H2 bulk fraction 
        bulk_he: He bulk fraction 

    """
    sort_pressure = np.argsort(reference_pressure)  # Sort ascending
    reference_pressure = reference_pressure[sort_pressure]
    reference_temperature = reference_temperature[sort_pressure]
    P0 = reference_pressure[-1]
    T0 = reference_temperature[-1]
    
    # Sort pressure ascending and create an inverse mask
    sort_pressure = np.argsort(pressure_grid) 
    use_pressure_grid = pressure_grid[sort_pressure]
    inv = np.zeros(len(pressure_grid), dtype=int)
    inv[sort_pressure] = np.arange(len(pressure_grid), dtype=int)

    start_t_grid = spi.interp1d(reference_pressure, reference_temperature, kind='linear', bounds_error=False, fill_value='extrapolate')(use_pressure_grid)
    temperature_grid = np.zeros(start_t_grid.shape)
    loc = np.argmin(abs(P0 - use_pressure_grid))
    temperature_grid[:loc] = start_t_grid[:loc]
    count = 0 
    while not np.allclose(start_t_grid, temperature_grid, rtol=1e-2): 
        count += 1 
        start_t_grid = temperature_grid.copy()
        cp = bulk_h2 * thermo.h2_normal_molar_heat_capacity(temperature_grid[loc:]) + bulk_he * thermo.he_molar_heat_capacity
        dTdp = spc.R * temperature_grid[loc:] / cp / use_pressure_grid[loc:] 
        temperature_grid[loc:] = T0 + sint.cumulative_trapezoid(dTdp, x=use_pressure_grid[loc:], initial=0)

    return temperature_grid[inv]


def hypsometric(pressure_grid, temperature_grid, planet_gravity, bulk_h2, bulk_he):
    """ Converts between pressure and height coordinates 
        Output is in kilometer units and referenced to the 1 bar pressure level

        pressure_grid: Pressure grid to work with, Pascals
        temperature_grid: Temperature grid to work with, K
        planet_gravity: Mean gravitational acceleration, m/s2 
        bulk_h2: H2 bulk fraction 
        bulk_he: He bulk fraction 

    """ 
    center_bin_pressure_grid = 0.5 * (pressure_grid[:-1] + pressure_grid[1:])
    M = bulk_h2 * thermo.h2_molar_mass * 1e-3 + bulk_he * thermo.he_molar_mass * 1e-3
    altitude_grid = sint.cumulative_trapezoid(spc.R * temperature_grid[:-1] / planet_gravity / M * np.log(pressure_grid[1:] / pressure_grid[:-1]), initial=0)
    altitude_1bar = spi.interp1d(center_bin_pressure_grid, altitude_grid)(1e5)
    altitude_grid = altitude_grid - altitude_1bar  # Subtract to set zero at the one bar level
    altitude_grid = spi.interp1d(center_bin_pressure_grid, altitude_grid, bounds_error=False, fill_value='extrapolate')(pressure_grid) / 1e3 

    return altitude_grid
    
def solar_to_deep_mole_fraction(x): 
    """ Mole fraction corresponding to solar nebula abundance
        Asplund 2009 
    """
    gases = {'H2O': x * 1.07e-3,   
             'PH3': x * 5.64e-7,
             'CH4': x * 5.9e-4,
             'NH3': x * 1.48e-4,
             'H2S': x * 2.89e-5} 
    return gases

def modify_dry_lapse(reference_temperature, reference_pressure, pressure_grid, bulk_h2, bulk_he, set_points=None, lapse_mods=None):
    """ Perform arbitrary adjustments to the dry adiabatic lapse rate

        Reference temperature and reference pressure are defined from occultation profiles 
        Pressure units are Pa 
        set_points is a list of 2-tuples which give bracketing pressures in bars 
        lapse_modes is a list which gives the modification to the dry adiabatic lapse rate for each bracket in set_points 
        If any of set_points falls above the occultation lower boundary, an error will be thrown 

    """ 
    
    P0 = reference_pressure[-1]
    loc = np.argmin(abs(P0 - pressure_grid))
    T0 = reference_temperature[-1]
    start_t_grid = spi.interp1d(reference_pressure, reference_temperature, kind='linear', bounds_error=False, fill_value='extrapolate')(pressure_grid)
    temperature_grid = np.zeros(start_t_grid.shape)
    temperature_grid[:loc] = start_t_grid[:loc]

    # Make slices to adjust lapse rates later 
    if set_points is not None:
        sp_slices = []
        for sp in set_points: 
            for i, p in enumerate(sp): 
                if p < P0: 
                    raise ValueError('Set point pressures must be below the occultation lower boundary')
                b = np.argmin(abs(p - pressure_grid[loc:]))
                if i == 0: 
                    start = b
                else: 
                    end = b
                    sp_slices.append(slice(start, end))
        
    count = 0 
    while not np.allclose(start_t_grid, temperature_grid, rtol=1e-2): 
        count += 1 
        start_t_grid = temperature_grid.copy()
        cp = bulk_h2 * thermo.h2_normal_molar_heat_capacity(temperature_grid[loc:]) + bulk_he * thermo.he_molar_heat_capacity
        dTdp = spc.R * temperature_grid[loc:] / cp / pressure_grid[loc:] 
    
        if set_points is not None:
            for i, sp in enumerate(sp_slices): 
                dTdp[sp] *= lapse_mods[i]
        
        temperature_grid[loc:] = T0 + sint.cumulative_trapezoid(dTdp, x=pressure_grid[loc:], initial=0)

  
    return temperature_grid 

def modify_developed_lapse(pressure_grid, temperature_grid, set_points=None, lapse_mods=None):
    """ Perform arbitrary adjustments to the dry adiabatic lapse rate

        Reference temperature and reference pressure are defined from occultation profiles 
        Pressure units are Pa 
        set_points is a list of 2-tuples which give bracketing pressures in bars 
        lapse_modes is a list which gives the modification to the dry adiabatic lapse rate for each bracket in set_points 
        If any of set_points falls above the occultation lower boundary, an error will be thrown 

    """ 
    
    # Make slices to adjust lapse rates later 
    if set_points is not None:
        sp_slices = []
        for sp in set_points: 
            for i, p in enumerate(sp): 
                b = np.argmin(abs(p - pressure_grid))
                if i == 0: 
                    start = b
                else: 
                    end = b
                    sp_slices.append(slice(start, end))
    
    dTdp = np.gradient(temperature_grid, pressure_grid) 
    if set_points is not None:
        for i, sp in enumerate(sp_slices): 
            dTdp[sp] *= lapse_mods[i]
    
    new_temperature_grid = temperature_grid[0] + sint.cumulative_trapezoid(dTdp, x=pressure_grid, initial=0)

    return new_temperature_grid 







