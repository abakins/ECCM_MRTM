import numpy as np
from numba import njit
from numba.core.types import float64, Tuple

# Type aliases
f64 = float64

# Constants
R = 8.314462618  # J/mol K
Pa_per_bar = 100000.0
Pa_per_mmHg = 133.32236842105263

# Phase index constants
N_PHASES = 2
SOLID = 0
LIQUID = 1

#########################
# Define gas thermodynamic and state properties
#########################
N_GASES = 5

#########################
# H2O
#########################
H2O_ID = 0
H2O_MOLAR_MASS = 18.0153  # g/mol
H2O_TRIPLE_POINT = 273.16  # K
H2O_MOLAR_HEAT_CAPACITY = 4 * R  # J/mol K


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def h2o_solid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat over ice/liquid.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-5631.1206, -22.179, 8.2312, -3.861e-2, 2.775e-5])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def h2o_liquid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat over ice/liquid.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-2313.0338, -177.848, 38.054, -0.13844, 7.4465e-5])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


#########################
# NH3
#########################
NH3_ID = 1
NH3_MOLAR_MASS = 17.0303  # g/mol
NH3_TRIPLE_POINT = 195.5  # K
NH3_MOLAR_HEAT_CAPACITY = 4.46 * R  # J/mol K


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def nh3_solid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for NH3.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-4122, 27.8632, -1.8163, 0, 0])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def nh3_liquid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for NH3.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-4409.3512, 63.0487, -8.4598, 5.51e-3, 6.8e-6])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


#########################
# H2S
#########################
H2S_ID = 2
H2S_MOLAR_MASS = 34.0809  # g/mol
H2S_TRIPLE_POINT = 187.61  # K
H2S_MOLAR_HEAT_CAPACITY = 4.01 * R  # J/mol K


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def h2s_solid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for H2S.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-2920.6, 14.156, 0, 0, 0])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def h2s_liquid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for H2S.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-2434.62, 11.4718, 0, 0, 0])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


#########################
# CH4
#########################
CH4_ID = 3
CH4_MOLAR_MASS = 16.04  # g/mol
CH4_TRIPLE_POINT = 90.7  # K
CH4_MOLAR_HEAT_CAPACITY = 4.5 * R  # J/mol K


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def ch4_solid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for CH4.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-1168.1, 10.710])
    lnp = a[0] / temperature + a[1]
    svp = np.exp(lnp)
    lh = -Z * R * a[0]
    return svp, lh


@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def ch4_liquid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for CH4.

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-1032.5, 9.216])
    lnp = a[0] / temperature + a[1]
    svp = np.exp(lnp)
    lh = -Z * R * a[0]
    return svp, lh


#########################
# PH3
#########################
PH3_ID = 4
PH3_MOLAR_MASS = 33.99758  # g/mol
PH3_TRIPLE_POINT = 139.41  # K
PH3_MOLAR_HEAT_CAPACITY = 4.5 * R  # J/mol K


# Vapor pressures and latent heats
@njit(
    Tuple((f64, f64))(f64, f64),
    cache=True,
)
def ph3_solid_saturation_vapor_pressure(temperature, Z=1.0):
    """Compute saturation vapor pressure and latent heat for PH3 (solid only).

    From D. DeBoer's Ph.D. thesis.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    Z: float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    """
    a = np.array([-1830, 9.8255, 0, 0, 0])
    lnp = (
        a[0] / temperature
        + a[1]
        + a[2] * np.log(temperature)
        + a[3] * temperature
        + a[4] * temperature**2
    )
    svp = np.exp(lnp)
    lh = (
        Z
        * R
        * (
            -a[0]
            + a[2] * temperature
            + a[3] * temperature**2
            + 2 * a[4] * temperature**3
        )
    )
    return svp, lh


#########################
# H2
#########################
H2_MOLAR_MASS = 2.01588  # g/mol
H2_TRIPLE_POINT = 13.81  # K


@njit(cache=True)
def h2_equilibrium_molar_heat_capacity(temperature):
    """Molar heat capacity of equilibrium H2 (from Farkas, interpolated).

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in K.

    Returns
    -------
    float or ndarray
        Molar heat capacity in J/(mol*K).
    """
    temps = np.array(
        [0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329]
    )
    a = np.array(
        [
            2.5,
            2.5014,
            2.6333,
            2.9628,
            3.4459,
            4.2345,
            4.5655,
            3.8721,
            3.3806,
            3.2115,
            3.1946,
            3.2402,
            3.3035,
            3.3630,
            3.411,
            3.4439,
            3.5,
        ]
    )
    molar_heat_capacity = np.interp(temperature, temps, a * R)
    return molar_heat_capacity


@njit(cache=True)
def h2_normal_molar_heat_capacity(temperature):
    """Molar heat capacity of normal H2 (from Trafton, interpolated).

    Parameters
    ----------
    temperature : float or ndarray
        Temperature in K.

    Returns
    -------
    float or ndarray
        Molar heat capacity in J/(mol*K).
    """
    temps = np.array(
        [0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329]
    )
    a = np.array(
        [
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5022,
            2.5154,
            2.6369,
            2.8138,
            2.9708,
            3.0976,
            3.2037,
            3.2899,
            3.3577,
            3.4085,
            3.4424,
            3.5,
        ]
    )
    molar_heat_capacity = np.interp(temperature, temps, a * R)
    return molar_heat_capacity


#########################
# He
#########################
HE_MOLAR_MASS = 4.0026  # g/mol
HE_TRIPLE_POINT = 1.76  # K
HE_MOLAR_HEAT_CAPACITY = 2.5 * R  # J/mol K


#########################
# Misc aerosol rules
#########################
NH4SH_MOLAR_MASS = 51.1074  # g/mol
NH4SH_LATENT_HEAT = 1.6e5  # J/mol


@njit(
    f64(
        f64,
    ),
    cache=True,
)
def h2o_nh3h2osolution_freezing_point(concentration):
    """Compute the depressed freezing point of water in an aqueous ammonia solution.

    Parameters
    ----------
    concentration : float
        NH3 volume mole fraction in the solution (dimensionless).

    Returns
    -------
    float
        Freezing point temperature in K.
    """
    Tf = (
        273.16
        - 124.167 * concentration
        + 189.963 * concentration**2
        - 2084.370 * concentration**3
    )
    return Tf


@njit(Tuple((f64, f64))(f64, f64, f64), cache=True)
def nh3_nh3h2osolution_saturation_vapor_pressure(temperature, concentration=0.0, Z=1.0):
    """Compute NH3 SVP and latent heat over an aqueous ammonia solution.

    From Briggs and Sackett (1989), with a multiplicative correction to
    enforce consistency with the pure NH3 SVP

    Parameters
    ----------
    temperature : float
        Temperature in K.
    concentration : float
        NH3 volume mole fraction in the solution.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        NH3 saturation vapor pressure in bar.
    lh : float
        Latent heat in J/mol.
    """

    r = (
        30.0048
        + 4.0134 * concentration * (concentration - 2)
        - (4949.75 + 2022.11 * concentration * (concentration - 2)) / temperature
    )
    svp = concentration * np.exp(r) * 1e-6  # Convert from dynes/cm**2 to bars
    lh = Z * R * (4949.75 + 2022.11 * concentration * (concentration - 2))

    # Correction: at c=1, match the pure NH3 liquid SVP
    # SVP_corrected = SVP_BS * (pure/BS_c1)^c
    # L_corrected = L_BS + c * (L_pure - L_BS_c1)
    r_c1 = (
        30.0048 + 4.0134 * (1.0 - 2.0) - (4949.75 + 2022.11 * (1.0 - 2.0)) / temperature
    )
    svp_bs_c1 = np.exp(r_c1) * 1e-6
    lh_bs_c1 = Z * R * (4949.75 + 2022.11 * (1.0 - 2.0))

    pure_svp, pure_lh = nh3_liquid_saturation_vapor_pressure(temperature, Z)

    if svp_bs_c1 > 0.0:
        ratio = pure_svp / svp_bs_c1
        svp = svp * ratio**concentration
        lh = lh + concentration * (pure_lh - lh_bs_c1)

    return svp, lh


@njit(Tuple((f64, f64))(f64, f64, f64), cache=True)
def h2o_nh3h2osolution_saturation_vapor_pressure(temperature, concentration=0.0, Z=1.0):
    """Compute H2O SVP and latent heat over an aqueous ammonia solution.

    From Briggs and Sackett (1989), with a multiplicative correction to
    enforce consistency with the pure H2O SVP (DeBoer) at c=0.

    Parameters
    ----------
    temperature : float
        Temperature in K.
    concentration : float
        NH3 volume mole fraction in the solution.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        H2O saturation vapor pressure in bar.
    lh : float
        Latent heat in J/mol.
    """
    # Briggs & Sackett base formula
    r = (
        29.0423
        + 4.0134 * concentration**2
        - (5540.48 + 2022.11 * concentration**2) / temperature
    )
    svp = (1 - concentration) * np.exp(r) * 1e-6  # Convert from dynes/cm**2 to bars
    lh = Z * R * (5540.48 + 2022.11 * concentration**2)

    # Correction: at c=0, match the pure H2O liquid SVP
    # SVP_corrected = SVP_BS * (pure/BS_c0)^(1-c)
    # L_corrected = L_BS + (1-c) * (L_pure - L_BS_c0)
    r_c0 = 29.0423 - 5540.48 / temperature
    svp_bs_c0 = np.exp(r_c0) * 1e-6
    lh_bs_c0 = Z * R * 5540.48

    pure_svp, pure_lh = h2o_liquid_saturation_vapor_pressure(temperature, Z)

    if svp_bs_c0 > 0.0:
        ratio = pure_svp / svp_bs_c0
        svp = svp * ratio ** (1.0 - concentration)
        lh = lh + (1.0 - concentration) * (pure_lh - lh_bs_c0)

    return svp, lh


MOLAR_MASS_ARRAY = np.array(
    [H2O_MOLAR_MASS, NH3_MOLAR_MASS, H2S_MOLAR_MASS, CH4_MOLAR_MASS, PH3_MOLAR_MASS]
)
MOLAR_HEAT_CAPACITY_ARRAY = np.array(
    [
        H2O_MOLAR_HEAT_CAPACITY,
        NH3_MOLAR_HEAT_CAPACITY,
        H2S_MOLAR_HEAT_CAPACITY,
        CH4_MOLAR_HEAT_CAPACITY,
        PH3_MOLAR_HEAT_CAPACITY,
    ]
)
TRIPLE_POINT_ARRAY = np.array(
    [
        H2O_TRIPLE_POINT,
        NH3_TRIPLE_POINT,
        H2S_TRIPLE_POINT,
        CH4_TRIPLE_POINT,
        PH3_TRIPLE_POINT,
    ]
)
GAS_INDEX = {"H2O": H2O_ID, "NH3": NH3_ID, "H2S": H2S_ID, "CH4": CH4_ID, "PH3": PH3_ID}
