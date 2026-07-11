from dataclasses import dataclass
from typing import Literal
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
import scipy.constants as spc
import scipy.interpolate as spi
import scipy.integrate as sint

from . import thermo
from .thermo import N_GASES, SOLID, LIQUID
from .core import run_eccm_core


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


@dataclass
class GasInput:
    """
    gas: Gas name string, e.g. "H2O", "NH3", "H2S", "CH4", "PH3"
    deep: Deep mole fraction.
    rh: Relative humidity fraction (default 1.0).
    """

    gas: str
    deep: float
    rh: float = 1.0


CLOUD_KWARGS_DEFAULTS = {
    "f_sed": 1.0,
    "T_eff": 124.0,
}


def run_eccm(
    pressure_grid: ArrayLike,
    reference_pressure: ArrayLike,
    reference_temperature: ArrayLike,
    planet_gravity: float,
    gases: list[GasInput],
    bulk_h2: float,
    bulk_he: float,
    latent_heat_update: bool = False,
    force_reference_above_pressure: float | None = 1.0 * spc.bar,
    use_compressibility: bool = False,
    h2_type: Literal["normal", "ortho", "para"] = "normal",
    cloud_model: Literal["equilibrium", "sediment"] = "equilibrium",
    cloud_kwargs: dict | None = None,
) -> dict:
    """Equilibrium cloud condensation model
    Inputs:
    pressure_grid: Pressure grid to work with, Pascals
    reference_pressure: Pressure points for reference_temperature, Pascals
    reference_temperature: Reference temperature profile, Kelvins
    planet_gravity: Planet gravity, m/s^2

    gases: list of GasInput objects specifying which gases to include
        Example:
            from molecule import H2O, NH3, H2S, CH4, PH3
            gases = [GasInput(H2O, deep=1e-3, rh=0.5),
                     GasInput(NH3, deep=1.5e-4),
                     GasInput(H2S, deep=3e-5)]
    bulk_h2: H2 bulk fraction
    bulk_he: He bulk fraction
    latent_heat_update: If True (default False), temperature profile updates will account for
                        latent heat of condensation
    force_reference_above_pressure: Pressure in Pa above which temperature profile is forced to
                                    match the reference profile, default is 1 bar level.
    use_compressibility: If True, will account for non-ideal gas effects
    cloud_model: 'equilibrium' (default) or 'sediment', controls how condensate is treated in the model
    cloud_kwargs: Parameters for the selected cloud model. For 'sediment':
        f_sed: Sedimentation efficiency parameter (default 1.0)
        T_eff: Effective temperature of the planet in K (default 124.0, for Jupiter)

    """

    pressure_grid = np.asarray(pressure_grid, dtype=np.float64)
    reference_pressure = np.asarray(reference_pressure, dtype=np.float64)
    reference_temperature = np.asarray(reference_temperature, dtype=np.float64)

    # Move from GasInput structure to
    # a flat array structure that works with Numba
    deep_x = np.zeros(N_GASES)
    rh = np.ones(N_GASES)
    gas_names = {}
    for g in gases:
        idx = thermo.GAS_INDEX[g.gas]
        deep_x[idx] = g.deep
        rh[idx] = g.rh
        gas_names[idx] = g.gas

    if cloud_model == "equilibrium":
        cloud_model_int = 0
    elif cloud_model == "sediment":
        cloud_model_int = 1
    else:
        raise ValueError("cloud_model must be 'equilibrium' or 'sediment'")

    h2_type_map = {"normal": 0, "ortho": 1, "para": 2}
    h2_type_int = h2_type_map.get(h2_type, 0)

    # Resolve cloud model parameters with defaults
    _ck = {**CLOUD_KWARGS_DEFAULTS, **(cloud_kwargs or {})}
    f_sed = float(_ck["f_sed"])
    T_eff = float(_ck["T_eff"])

    # Sort so that first index is deepest pressure
    pressure_grid = pressure_grid[np.argsort(pressure_grid)[::-1]]

    # Starting from a reference temperature profile, extrapolate to the deepest pressure
    # using the dry adiabatic lapse rate
    temperature_grid = compute_temperature_guess(
        pressure_grid, reference_temperature, reference_pressure, bulk_h2, bulk_he
    )

    # Run the ECCM
    new_temperature_grid, x, aerosol, a_nh4sh, a_solution, c_nh3h2o, z_profile = (
        run_eccm_core(
            pressure_grid,
            temperature_grid,
            deep_x,
            rh,
            bulk_h2,
            bulk_he,
            latent_heat_update,
            False,  # No forced reference temperature
            use_compressibility,
            cloud_model_int,
            f_sed,
            T_eff,
            planet_gravity,
            h2_type_int,
        )
    )

    # Loop to force temperature match to reference
    if force_reference_above_pressure is not None:
        force_index = np.argmin(abs(pressure_grid - force_reference_above_pressure))
        ref_t_grid_func = spi.interp1d(
            reference_pressure,
            reference_temperature,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        offs = new_temperature_grid[force_index] - ref_t_grid_func(
            pressure_grid[force_index]
        )

        off_count = 0
        while abs(offs) > 1e-2:
            off_count = off_count + 1
            if off_count > 250:
                print("""Temperature profile matching iteration is either diverging or oscillating,
                         printing current offset and moving ahead anyway""")
                print(offs)
                break
            new_temperature_grid[0] = new_temperature_grid[0] - offs
            (
                new_temperature_grid,
                x,
                aerosol,
                a_nh4sh,
                a_solution,
                c_nh3h2o,
                z_profile,
            ) = run_eccm_core(
                pressure_grid,
                new_temperature_grid,
                deep_x,
                rh,
                bulk_h2,
                bulk_he,
                latent_heat_update,
                False,
                use_compressibility,
                cloud_model_int,
                f_sed,
                T_eff,
                planet_gravity,
                h2_type_int,
            )
            offs = new_temperature_grid[force_index] - ref_t_grid_func(
                pressure_grid[force_index]
            )
        new_temperature_grid[force_index:] = ref_t_grid_func(
            pressure_grid[force_index:]
        )

        # And finally run once more, forcing the temperature profile
        new_temperature_grid, x, aerosol, a_nh4sh, a_solution, c_nh3h2o, z_profile = (
            run_eccm_core(
                pressure_grid,
                new_temperature_grid,
                deep_x,
                rh,
                bulk_h2,
                bulk_he,
                latent_heat_update,
                True,
                use_compressibility,
                cloud_model_int,
                f_sed,
                T_eff,
                planet_gravity,
                h2_type_int,
            )
        )

    # Adjust H2/He to maintain the appropriate split
    total_x = np.sum(x, axis=0)
    ratio_h2 = bulk_h2 / (1 - bulk_h2)
    x_h2 = (1 - total_x) / (1 + 1 / ratio_h2)
    ratio_he = bulk_he / (1 - bulk_he)
    x_he = (1 - total_x) / (1 + 1 / ratio_he)

    # Compute altitude/pressure mapping
    altitude_grid = hypsometric(
        pressure_grid,
        new_temperature_grid,
        planet_gravity,
        x_h2,
        x_he,
        x,
        z_profile,
    )

    # Package results
    gas_profiles = {}
    aerosol_densities = {}
    for idx, gas_name in gas_names.items():
        gas_profiles[gas_name] = x[idx]
        aerosol_densities[gas_name] = {
            "solid": aerosol[idx, SOLID],
            "liquid": aerosol[idx, LIQUID],
        }
    aerosol_densities["NH4SH"] = {"solid": a_nh4sh, "liquid": np.zeros_like(a_nh4sh)}
    aerosol_densities["H2O_NH3_SOLUTION"] = {
        "solid": np.zeros_like(a_solution),
        "liquid": a_solution,
    }

    result = dict(
        pressure=pressure_grid,
        temperature=new_temperature_grid,
        altitude=altitude_grid,
        gas_profiles=gas_profiles,
        aerosol_densities=aerosol_densities,
        nh3h2o_concentration=c_nh3h2o,
        mole_fraction_h2=x_h2,
        mole_fraction_he=x_he,
        compressibility_profile=z_profile,
    )

    return result


def compute_temperature_guess(
    pressure_grid: ArrayLike,
    reference_temperature: ArrayLike,
    reference_pressure: ArrayLike,
    bulk_h2: float,
    bulk_he: float,
) -> npt.NDArray[np.floating]:
    """Extrapolates a reference temperature profile assuming the dry adiabatic lapse rate

    pressure_grid: Pressure grid to work with, Pascals
    reference_temperature: Reference temperature profile, from occultations, Kelvins
    reference_pressure: Pressure points for reference_temperature, Pascals
    bulk_h2: H2 bulk fraction
    bulk_he: He bulk fraction

    """
    pressure_grid = np.asarray(pressure_grid, dtype=np.float64)
    reference_pressure = np.asarray(reference_pressure, dtype=np.float64)
    reference_temperature = np.asarray(reference_temperature, dtype=np.float64)

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

    start_t_grid = spi.interp1d(
        reference_pressure,
        reference_temperature,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(use_pressure_grid)
    temperature_grid = np.zeros(start_t_grid.shape)
    loc = np.argmin(abs(P0 - use_pressure_grid))
    temperature_grid[:loc] = start_t_grid[:loc]
    count = 0
    while not np.allclose(start_t_grid, temperature_grid, rtol=1e-2):
        count += 1
        start_t_grid = temperature_grid.copy()
        cp = (
            bulk_h2 * thermo.h2_normal_molar_heat_capacity(temperature_grid[loc:])
            + bulk_he * thermo.HE_MOLAR_HEAT_CAPACITY
        )
        dTdp = spc.R * temperature_grid[loc:] / cp / use_pressure_grid[loc:]
        temperature_grid[loc:] = T0 + sint.cumulative_trapezoid(
            dTdp, x=use_pressure_grid[loc:], initial=0
        )

    return temperature_grid[inv]


def hypsometric(
    pressure_grid: ArrayLike,
    temperature_grid: ArrayLike,
    planet_gravity: float,
    x_h2: ArrayLike,
    x_he: ArrayLike,
    x: ArrayLike,
    z_profile: ArrayLike,
) -> npt.NDArray[np.floating]:
    """Converts between pressure and height coordinates
    Output is in kilometer units and referenced to the 1 bar pressure level

    pressure_grid: Pressure grid to work with, Pascals
    temperature_grid: Temperature grid to work with, K
    planet_gravity: Mean gravitational acceleration, m/s2
    x_h2: H2 mole fraction profile
    x_he: He mole fraction profile
    x: Other gas mole fraction profiles
    z_profile: Compressibility factor profile (Z=1 for ideal gas)

    """
    pressure_grid = np.asarray(pressure_grid, dtype=np.float64)
    temperature_grid = np.asarray(temperature_grid, dtype=np.float64)
    x_h2 = np.asarray(x_h2, dtype=np.float64)
    x_he = np.asarray(x_he, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z_profile = np.asarray(z_profile, dtype=np.float64)

    center_bin_pressure_grid = 0.5 * (pressure_grid[:-1] + pressure_grid[1:])
    M = (
        x_h2 * thermo.H2_MOLAR_MASS * 1e-3
        + x_he * thermo.HE_MOLAR_MASS * 1e-3
        + x[thermo.H2O_ID] * thermo.H2O_MOLAR_MASS * 1e-3
        + x[thermo.CH4_ID] * thermo.CH4_MOLAR_MASS * 1e-3
    )
    altitude_grid = sint.cumulative_trapezoid(
        z_profile[:-1]
        * spc.R
        * temperature_grid[:-1]
        / planet_gravity
        / M[:-1]
        * np.log(pressure_grid[1:] / pressure_grid[:-1]),
        initial=0,
    )
    altitude_1bar = spi.interp1d(center_bin_pressure_grid, altitude_grid)(1e5)
    altitude_grid = (
        altitude_grid - altitude_1bar
    )  # Subtract to set zero at the one bar level
    altitude_grid = (
        spi.interp1d(
            center_bin_pressure_grid,
            altitude_grid,
            bounds_error=False,
            fill_value="extrapolate",
        )(pressure_grid)
        / 1e3
    )

    return altitude_grid


def solar_concentration(
    model: Literal["asplund2009", "asplund2021", "lodders2025"] = "asplund2009",
) -> dict[str, float]:
    """Solar photosphere abundances of heavy elements
    Returned as a ratio of X/H2
    See B. Karpowicz thesis

    Note:
    There are apparently significant variations
    in "protosolar" extrapolations for the contemporary photosphere.
    Lodders values are much higher than Asplund as a result.
    Default to the mid-point, Asplund 2009
    """
    if model.lower() == "asplund2009":
        # Bulk abundance is est. at 0.04 dex (0.05 He) higher than solar photosphere
        keys = [
            "H2",
            "He",
            "C",
            "N",
            "O",
            "P",
            "S",
        ]
        mole_fraction = np.array(
            [
                0.8380249589278793,
                0.16006152453540878,
                0.0004946373981929341,
                0.00012424729690380356,
                0.0009000920981922898,
                4.7237504804318617e-07,
                2.4226292090241593e-5,
            ]
        )
        concentration = mole_fraction / mole_fraction[0]
        concentration = zip(keys, concentration)

    elif model.lower() == "asplund2021":
        # Bulk abundance is est. at 0.03 dex (0.035 He) higher than solar photosphere
        # These could be twice as high (see their paper), but we'll follow a convention
        # here of having Asplund/Grevesse values be lower than Lodders
        keys = [
            "H2",
            "He",
            "C",
            "N",
            "O",
            "P",
            "S",
        ]
        mole_fraction = np.array(
            [
                0.8485152095052825,
                0.14951667381295772,
                0.0005244325352704977,
                0.0001229389860872382,
                0.0008906142241675396,
                4.674009890476898e-7,
                2.3971191812193094e-5,
            ]
        )
        concentration = mole_fraction / mole_fraction[0]
        concentration = zip(keys, concentration)

    elif model.lower() == "lodders2025":
        # Bulk abundance is est. at 0.088 dex (0.07 He) higher than solar photosphere
        keys = [
            "H2",
            "He",
            "C",
            "N",
            "O",
            "P",
            "S",
        ]
        mole_fraction = np.array(
            [
                0.8337075110728833,
                0.16369812681522317,
                0.0006607599472639128,
                0.0001950040852976857,
                0.001175015809197821,
                5.623979252369145e-07,
                2.9515062679332642e-05,
            ]
        )
        concentration = mole_fraction / mole_fraction[0]
        concentration = zip(keys, concentration)
    else:
        raise ValueError("Unsupported model for solar_concentration")
    return dict(concentration)


def modify_dry_lapse(
    reference_temperature: ArrayLike,
    reference_pressure: ArrayLike,
    pressure_grid: ArrayLike,
    bulk_h2: float,
    bulk_he: float,
    set_points: list[tuple[float, float]] | None = None,
    lapse_mods: list[float] | None = None,
) -> npt.NDArray[np.floating]:
    """Perform arbitrary adjustments to the dry adiabatic lapse rate

    Reference temperature and reference pressure are defined from occultation profiles
    Pressure units are Pa
    set_points is a list of 2-tuples which give bracketing pressures in bars
    lapse_modes is a list which gives the modification to the dry adiabatic lapse rate for each bracket in set_points
    If any of set_points falls above the occultation lower boundary, an error will be thrown

    """

    reference_temperature = np.asarray(reference_temperature, dtype=np.float64)
    reference_pressure = np.asarray(reference_pressure, dtype=np.float64)
    pressure_grid = np.asarray(pressure_grid, dtype=np.float64)

    P0 = reference_pressure[-1]
    loc = np.argmin(abs(P0 - pressure_grid))
    T0 = reference_temperature[-1]
    start_t_grid = spi.interp1d(
        reference_pressure,
        reference_temperature,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(pressure_grid)
    temperature_grid = np.zeros(start_t_grid.shape)
    temperature_grid[:loc] = start_t_grid[:loc]

    # Make slices to adjust lapse rates later
    if set_points is not None:
        sp_slices = []
        for sp in set_points:
            for i, p in enumerate(sp):
                if p < P0:
                    raise ValueError(
                        "Set point pressures must be below the occultation lower boundary"
                    )
                b = np.argmin(abs(p - pressure_grid[loc:]))
                if i == 0:
                    start = b
                else:
                    end = b
                    if start > end:
                        start, end = end, start
                    sp_slices.append(slice(start, end))

    count = 0
    while not np.allclose(start_t_grid, temperature_grid, rtol=1e-2):
        count += 1
        start_t_grid = temperature_grid.copy()
        cp = (
            bulk_h2 * thermo.h2_normal_molar_heat_capacity(temperature_grid[loc:])
            + bulk_he * thermo.HE_MOLAR_HEAT_CAPACITY
        )
        dTdp = spc.R * temperature_grid[loc:] / cp / pressure_grid[loc:]

        if set_points is not None:
            for i, sp in enumerate(sp_slices):
                dTdp[sp] *= lapse_mods[i]

        temperature_grid[loc:] = T0 + sint.cumulative_trapezoid(
            dTdp, x=pressure_grid[loc:], initial=0
        )

    return temperature_grid


def modify_developed_lapse(
    pressure_grid: ArrayLike,
    temperature_grid: ArrayLike,
    set_points: list[tuple[float, float]] | None = None,
    lapse_mods: list[float] | None = None,
) -> npt.NDArray[np.floating]:
    """Perform arbitrary adjustments to the dry adiabatic lapse rate

    Reference temperature and reference pressure are defined from occultation profiles, see data/ro_profiles
    Pressure units are Pa
    set_points is a list of 2-tuples which give bracketing pressures in bars
    lapse_modes is a list which gives the modification to the dry adiabatic lapse rate for each bracket in set_points
    If any of set_points falls above the occultation lower boundary, an error will be thrown

    """

    pressure_grid = np.asarray(pressure_grid, dtype=np.float64)
    temperature_grid = np.asarray(temperature_grid, dtype=np.float64)

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
                    if start > end:
                        start, end = end, start
                    sp_slices.append(slice(start, end))

    dTdp = np.gradient(temperature_grid, pressure_grid)
    if set_points is not None:
        for i, sp in enumerate(sp_slices):
            dTdp[sp] *= lapse_mods[i]

    new_temperature_grid = temperature_grid[0] + sint.cumulative_trapezoid(
        dTdp, x=pressure_grid, initial=0
    )

    return new_temperature_grid
