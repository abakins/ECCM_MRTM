import numpy as np 
from numba import njit 

# Constants 
R = 8.314462618 # J/mol K
Pa_per_bar = 100000.0
Pa_per_mmHg = 133.32236842105263

h2_molar_mass = 2.01588  # g/mol 
he_molar_mass = 4.0026  
ch4_molar_mass = 16.04  
h2o_molar_mass = 18.0153   
nh3_molar_mass = 17.0303   
ph3_molar_mass = 33.99758 
h2s_molar_mass  = 34.0809  
nh4sh_molar_mass= 51.11

h2o_triple_point = 273.16 # Kelvin
ch4_triple_point = 90.7  
h2s_triple_point = 187.61
nh3_triple_point = 195.5
ph3_triple_point = 139.41
h2_triple_point = 13.81 
he_triple_point = 1.76  

nh4sh_latent_heat = 1.6e5  # J/mol

# Several molar heat capacities could stand to be upgraded from Briggs and Sackett 1989, or newer
h2s_molar_heat_capacity = 4.01 * R  # J/mol K
h2o_molar_heat_capacity = 4 * R 
nh3_molar_heat_capacity = 4.46 * R  
he_molar_heat_capacity = 2.5 * R
ch4_molar_heat_capacity = 4.5 * R

@njit
def h2_equilibrium_molar_heat_capacity(temperature):
    """ Input is temperature in K 
        For equilibrium H2 from Farkas 
    """
    temps = np.array([0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329])
    a = np.array([2.5, 2.5014, 2.6333, 2.9628, 3.4459, 4.2345, 4.5655, 3.8721, 3.3806, 3.2115, 3.1946, 3.2402, 3.3035, 3.3630, 3.411, 3.4439, 3.5])
    molar_heat_capacity = np.interp(temperature, temps, a * R)
    return molar_heat_capacity

@njit
def h2_normal_molar_heat_capacity(temperature): 
    """ Input is temperature in K 
        For normal H2 from Trafton
    """
    temps = np.array([0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329])
    a = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5022, 2.5154, 2.6369, 2.8138, 2.9708, 3.0976, 3.2037, 3.2899, 3.3577, 3.4085, 3.4424, 3.5])
    molar_heat_capacity = np.interp(temperature, temps, a * R)
    return molar_heat_capacity

@njit
def h2o_nh3h2osolution_freezing_point(concentration): 
    """ Determines the depressed freezing point of water in an aqueous ammonia solution 
        Concentration is volume mole fraction of ammonia in solution"""
    Tf = 273.16 - 124.167*concentration + 189.963 * concentration**2 - 2084.370*concentration**3
    return Tf 



# Vapor pressures and latent heats 
@njit
def ph3_solid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of phosphine phase transitions 
        From D. DeBoer's Ph.D. thesis
        temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-1830, 9.8255, 0, 0, 0])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def h2s_solid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of hydrogen sulfide phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-2920.6, 14.156, 0, 0, 0])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def h2s_liquid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of hydrogen sulfide phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-2434.62, 11.4718, 0, 0, 0])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def h2o_solid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of water phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-5631.1206, -22.179, 8.2312, -3.861e-2, 2.775e-5])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def h2o_liquid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of water phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-2313.0338, -177.848, 38.054, -0.13844, 7.4465e-5])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh
        
@njit
def nh3_solid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of ammonia phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-4122, 27.8632, -1.8163, 0, 0])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def nh3_liquid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of ammonia phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-4409.3512, 63.0487, -8.4598, 5.51e-3, 6.8e-6])
    lnp = a[0] / temperature + a[1] + a[2] * np.log(temperature) + a[3] * temperature + a[4] * temperature**2
    svp = np.exp(lnp)
    lh = R * (-a[0] + a[2] * temperature + a[3] * temperature**2 + 2 * a[4] * temperature**3)
    return svp, lh

@njit
def ch4_solid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of methane phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-1168.1, 10.710])
    lnp = a[0] / temperature + a[1]
    svp = np.exp(lnp)
    lh = -R * a[0]
    return svp, lh 

@njit
def ch4_liquid_saturation_vapor_pressure(temperature): 
    """ Computes saturation vapor pressure and latent heat of methane phase transitions 
        From D. DeBoer's Ph.D. thesis
        :param temperature: Temperature in Kelvin
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    a = np.array([-1032.5, 9.216])
    lnp = a[0] / temperature + a[1]
    svp = np.exp(lnp)
    lh = -R * a[0]
    return svp, lh 

@njit
def nh3_nh3h2osolution_saturation_vapor_pressure(temperature, concentration=0.): 
    """ Computes saturation vapor pressure and latent heat of ammonia phase transitions 
        in an aqueous ammonia solution
        From Briggs and Sackett 1989
        :param temperature: Temperature in Kelvin
        :param concentration: Ammonia solution volume mole fraction 
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    # Concentration is ammonia volume mole fraction 
    r = 30.0048 + 4.0134*concentration*(concentration-2) - (4949.75 + 2022.11*concentration*(concentration-2))/temperature
    svp = concentration * np.exp(r) * 1e-6 # Convert from dynes/cm**2 to bars 
    lh = R * (4949.75 + 2022.11*concentration*(concentration-2))
    return svp, lh   

@njit
def h2o_nh3h2osolution_saturation_vapor_pressure(temperature, concentration=0.): 
    """ Computes saturation vapor pressure and latent heat of water phase transitions 
        in an aqueous ammonia solution
        From Briggs and Sackett 1989
        :param temperature: Temperature in Kelvin
        :param concentration: Ammonia solution volume mole fraction 
        :return: Saturation vapor pressure in bars and latent heat in J/mol
    """
    # Concentration is ammonia volume mole fraction 
    r = 29.0423 + 4.0134*concentration**2 - (5540.48 + 2022.11*concentration**2)/temperature
    svp = (1-concentration) * np.exp(r) * 1e-6 # Convert from dynes/cm**2 to bars  
    lh = R * (5540.48 + 2022.11*concentration**2)
    return svp, lh   
