import numpy as np
import scipy.constants as spc
from numba import njit
from numba.core.types import float64, int64, boolean, Array, Tuple

from . import thermo
from ..utils import find_root
from .eos import compute_Z, compute_Cp

# Type aliases
f64 = float64
i64 = int64
b1 = boolean
arr1d = Array(float64, 1, "A")  # 1D float64 array (any layout, accepts slices)
arr1dc = Array(float64, 1, "C")  # 1D float64 array (C-contiguous)
arr2dc = Array(float64, 2, "C")  # 2D float64 array (C-contiguous)
arr3dc = Array(float64, 3, "C")  # 3D float64 array (C-contiguous)


@njit(Tuple((f64, f64, i64))(i64, f64, f64), cache=True)
def svp_dispatch(gas_id, T, Z):
    """Dispatch to the saturation vapor pressure function for a given gas.

    Selects solid or liquid SVP based on whether T is below or above
    the triple point for the specified gas.

    Parameters
    ----------
    gas_id : int
        Gas identifier (e.g. thermo.H2O_ID, thermo.NH3_ID).
    T : float
        Temperature in K.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    svp : float
        Saturation vapor pressure in bar.
    lh : float
        Latent heat of phase transition in J/mol.
    phase : int
        Phase flag (thermo.SOLID or thermo.LIQUID).
    """
    tp = thermo.TRIPLE_POINT_ARRAY[gas_id]
    if T < tp:
        phase = thermo.SOLID
        if gas_id == thermo.H2O_ID:
            svp, lh = thermo.h2o_solid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.NH3_ID:
            svp, lh = thermo.nh3_solid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.H2S_ID:
            svp, lh = thermo.h2s_solid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.CH4_ID:
            svp, lh = thermo.ch4_solid_saturation_vapor_pressure(T, Z)
        else:
            svp, lh = thermo.ph3_solid_saturation_vapor_pressure(T, Z)
    else:
        phase = thermo.LIQUID
        if gas_id == thermo.H2O_ID:
            svp, lh = thermo.h2o_liquid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.NH3_ID:
            svp, lh = thermo.nh3_liquid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.H2S_ID:
            svp, lh = thermo.h2s_liquid_saturation_vapor_pressure(T, Z)
        elif gas_id == thermo.CH4_ID:
            svp, lh = thermo.ch4_liquid_saturation_vapor_pressure(T, Z)
        else:
            # PH3 has no liquid phase in model — use solid
            svp, lh = thermo.ph3_solid_saturation_vapor_pressure(T, Z)
            phase = thermo.SOLID
    return svp, lh, phase


@njit(Tuple((f64, f64, i64))(f64, f64, f64, f64, f64, i64, f64), cache=True)
def update_single_gas(p2, T2, dT, dp, x1, gas_id, Z):
    """Compute Clausius-Clapeyron condensation for a single gas

    d ln X_i = L_i/(RT)*d ln T - d ln P
    Parameters
    ----------
    p2 : float
        Pressure at the current level in Pa.
    T2 : float
        Temperature at the current level in K.
    dT : float
        Temperature step (T2 - T1) in K.
    dp : float
        Pressure step (p2 - p1) in Pa.
    x1 : float
        Mole fraction of the gas at the previous level.
    gas_id : int
        Gas identifier index.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    x2 : float
        Updated mole fraction after condensation.
    lh : float
        Latent heat used in J/mol (0 if no condensation).
    phase : int
        Phase of condensate (thermo.SOLID or thermo.LIQUID).
    """
    svp, lh, phase = svp_dispatch(gas_id, T2, Z)
    svp_pa = svp * 1e5  # bar to Pa
    if (svp_pa / p2) < x1:
        dlnT = dT / T2
        dlnP = dp / p2
        dx = x1 * lh / (Z * spc.R * T2) * dlnT - x1 * dlnP
        dx = max(dx, (svp_pa / p2) - x1)
    else:
        dx = 0.0
        lh = 0.0
    x2 = x1 + dx
    return x2, lh, phase


@njit(f64(f64, f64, f64, f64, f64, f64, f64), cache=True)
def update_cloud(p1, p2, T2, x1, x2, molar_mass, Z):
    """Convert mole fraction change to cloud mass density (Weidenschilling & Lewis, 1973).

    aer = M * dx * P^2 / (Z * R * T * dP)

    Parameters
    ----------
    p1 : float
        Pressure at previous level in Pa.
    p2 : float
        Pressure at current level in Pa.
    T2 : float
        Temperature at current level in K.
    x1 : float
        Mole fraction at previous level.
    x2 : float
        Mole fraction at current level.
    molar_mass : float
        Molar mass of the condensate in g/mol.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    float
        Aerosol mass density in g/m^3.
    """
    dx = x2 - x1
    dp = p2 - p1
    aer = molar_mass * dx * p2**2 / (Z * spc.R * T2 * dp)  # Units are g/m3
    return aer


@njit(f64(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64), cache=True)
def compute_cloud_sediment(
    T,
    P,
    T_eff,
    H,
    gamma_wet,
    gamma_dry,
    dz,
    q_v,
    q_v_prev,
    q_c_prev,
    f_sed,
    mu,
    cp,
):
    """Compute cloud condensate mixing ratio via diffusion/sedimentation balance.

    Discrete solution of Ackerman & Marley (2001) Eq. 4:
    q_c = -K*(q_v - q_v_prev - q_c_prev) / (K + f_sed*w*dz)

    Due to Ackerman & Marley 2001; implementation adapted from
    Bryan Karpowicz's LRTM repository (https://github.com/karpob/lrtm).

    Parameters
    ----------
    T : float
        Temperature in K.
    P : float
        Pressure in Pa.
    T_eff : float
        Planet effective temperature in K (sets thermal flux).
    H : float
        Atmospheric scale height in m.
    gamma_wet : float
        Wet adiabatic lapse rate in K/m.
    gamma_dry : float
        Dry adiabatic lapse rate in K/m.
    dz : float
        Layer thickness in m (positive upward).
    q_v : float
        Vapor mole fraction at current level.
    q_v_prev : float
        Vapor mole fraction at previous level.
    q_c_prev : float
        Cloud condensate mixing ratio at previous level (mol/mol).
    f_sed : float
        Sedimentation efficiency parameter.
    mu : float
        Mean molecular weight in g/mol.
    cp : float
        Specific heat capacity in J/(K*mol).

    Returns
    -------
    float
        Cloud condensate mixing ratio q_c (mol/mol).
    """
    # Convert to CGS units used internally
    P_bars = P / 1e5  # Pa to bar
    H_cm = H * 100.0  # m to cm
    gamma_wet_cgs = gamma_wet / 100.0  # K/m to K/cm
    gamma_dry_cgs = gamma_dry / 100.0  # K/m to K/cm
    dz_cm = dz * 100.0  # m to cm
    cp_cgs = cp * 1e7  # J/(K*mol) to erg/(K*mol)

    R_cgs = 8.3143e7  # erg/(mol*K)

    # Lapse rate ratio (bounded below by 0.1)
    if gamma_dry_cgs != 0:
        gamma_ratio = max(gamma_wet_cgs / gamma_dry_cgs, 0.1)
    else:
        gamma_ratio = 0.1

    # Mixing length
    L_mix = H_cm * gamma_ratio

    # Density (g/cm³): P in dyne/cm² = bars * 1e6
    P_cgs = P_bars * 1.0e6  # dyne/cm²
    rho = mu * P_cgs / (R_cgs * T)

    # Thermal flux (erg/cm²/s)
    sigma = 5.6704e-5  # erg/(cm²*s*K⁴) Stefan-Boltzmann
    Flux = sigma * T_eff**4

    # Eddy diffusion coefficient (Eq. 5, Ackerman & Marley 2001)
    if gamma_ratio > 1.0:
        F_factor = (gamma_ratio - 1.0) / gamma_dry_cgs
    else:
        F_factor = 0.0

    if F_factor > 0 and rho > 0 and cp_cgs > 0:
        K = (
            H_cm
            * F_factor
            * (L_mix / H_cm) ** (4.0 / 3.0)
            * (1e-3 * R_cgs * Flux / (mu * rho * cp_cgs)) ** (1.0 / 3.0)
        )
    else:
        K = 1.0e5  # minimum

    K = max(K, 1.0e5)  # lower bound on eddy diffusion

    # Velocity scale
    wstar = K / L_mix

    # Discrete diffusion/sedimentation solution
    numerator = -K * (q_v - q_v_prev - q_c_prev)
    denominator = K + f_sed * wstar * abs(dz_cm)

    if denominator > 0:
        q_c = numerator / denominator
    else:
        q_c = 0.0

    return q_c


# NH4SH reaction
@njit(Tuple((f64, f64, f64))(f64, f64, f64, f64, f64, f64), cache=True)
def update_nh4sh(p1, T1, p2, T2, x1_h2s, x1_nh3):
    """Compute NH4SH reaction equilibrium for one pressure step.

    Checks whether the product of NH3 and H2S partial pressures exceeds
    the equilibrium constant K = exp(34.150 - 10834/T). If so, computes
    the mole fraction change and enforces the equilibrium constraint.

    Parameters
    ----------
    p1 : float
        Pressure at previous level in Pa.
    T1 : float
        Temperature at previous level in K.
    p2 : float
        Pressure at current level in Pa.
    T2 : float
        Temperature at current level in K.
    x1_h2s : float
        H2S mole fraction before reaction.
    x1_nh3 : float
        NH3 mole fraction before reaction.

    Returns
    -------
    x2_h2s : float
        H2S mole fraction after reaction.
    x2_nh3 : float
        NH3 mole fraction after reaction.
    lh : float
        Latent heat released by reaction in J/mol (0 if no reaction).
    """
    dT = T2 - T1
    dp = p2 - p1
    nh3p = p2 / spc.bar * x1_nh3
    h2sp = p2 / spc.bar * x1_h2s
    eqr = 34.150 - 10834 / T2  # Aerosol system equilibrium constant
    if np.log(nh3p * h2sp) > eqr:
        dx = (x1_h2s * x1_nh3) / (x1_nh3 + x1_h2s) * (10834 * dT / T2**2 - 2 * dp / p2)
        lh = thermo.NH4SH_LATENT_HEAT
        s1, s2 = find_root.solve_quadratic(
            1.0, x1_h2s + x1_nh3, x1_h2s * x1_nh3 - np.exp(eqr) / (p2 / spc.bar) ** 2
        )
        dxr = max(s1, s2)
        dx = max(dx, dxr)
    else:
        dx = 0.0
        lh = 0.0

    x2_h2s = x1_h2s + dx
    x2_nh3 = x1_nh3 + dx

    return x2_h2s, x2_nh3, lh


# Utilities for NH3/H2O solution concentration finding
@njit(f64(f64, f64, f64, f64), cache=True)
def fn_nh3(gc, pnh3, T1, Z):
    svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T1, gc, Z)[0] * 1e5
    return abs(svp_nh3 - pnh3)


@njit(f64(f64, f64, f64, f64), cache=True)
def fn_h2o(gc, ph2o, T1, Z):
    svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T1, gc, Z)[0] * 1e5
    return abs(svp_h2o - ph2o)


@njit(f64(f64, f64, f64, f64, f64), cache=True)
def fn_both(gc, pnh3, ph2o, T1, Z):
    if gc < 0:
        gc = 0.0
    elif gc > 1:
        gc = 1.0
    svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T1, gc, Z)[0] * 1e5
    svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T1, gc, Z)[0] * 1e5
    return abs((1.0 - gc) * (svp_nh3 - pnh3) - gc * (svp_h2o - ph2o))


# NH3-H2O solution condensation
@njit(Tuple((f64, f64, f64, f64, f64, b1))(f64, f64, f64, f64, f64, f64, f64, f64, f64))
def update_nh3_h2o_solution(p1, T1, p2, T2, x1_h2o, x1_nh3, c1_nh3h2o, x_dry_air, Z):
    dT = T2 - T1
    dp = p2 - p1

    # Check the vapor pressures, does solution condensation occur?
    Tf = thermo.h2o_nh3h2osolution_freezing_point(c1_nh3h2o)
    if T2 < Tf:
        # Water is freezing, solution condensation does not occur
        # Pass through
        x2_h2o = x2_nh3 = lh_h2o = lh_nh3 = 0
        c2_nh3h2o = c1_nh3h2o
        solution_flag = False
    else:
        # Solution condensation might occur
        pnh3 = x1_nh3 * p2
        ph2o = x1_h2o * p2

        # Checking bracketing concentrations for pure constituent
        svp_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(T2, 1.0, Z)
        svp_nh3 = svp_nh3[0] * 1e5  # Solution SVP, to Pa
        if pnh3 >= svp_nh3:
            c_max = 1.0
        else:
            c_max = find_root.brent(fn_nh3, 0.0, 1.0, args=(pnh3, T2, Z))[0]

        svp_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(T2, 0.0, Z)
        svp_h2o = svp_h2o[0] * 1e5
        if ph2o >= svp_h2o:
            c_min = 0.0
        else:
            c_min = find_root.brent(fn_h2o, 0.0, 1.0, args=(ph2o, T2, Z))[0]
        if c_max < c_min:
            c_min, c_max = c_max, c_min

        # c_min can't be exactly zero or one, or else bracketing breaks
        c = find_root.brent(
            fn_both, c_min + 1e-6, c_max - 1e-6, args=(pnh3, ph2o, T2, Z)
        )[0]
        eval_c = fn_both(c, pnh3, ph2o, T2, Z)
        eval_allh2o = fn_both(0, pnh3, ph2o, T2, Z)
        eval_allnh3 = fn_both(1, pnh3, ph2o, T2, Z)

        if (eval_allh2o < eval_allnh3) and (eval_allh2o < eval_c):
            conc = 0.0
        elif (eval_allnh3 < eval_allh2o) and (eval_allnh3 < eval_c):
            conc = 1.0
        else:
            conc = c

        if np.isclose(conc, 0.0) | np.isclose(conc, 1.0):
            # No good solution
            # Pass through
            x2_h2o = x2_nh3 = lh_h2o = lh_nh3 = 0
            c2_nh3h2o = c1_nh3h2o
            solution_flag = False
        else:
            # A concentration was found

            svp_h2o, lh_h2o = thermo.h2o_nh3h2osolution_saturation_vapor_pressure(
                T2, conc, Z
            )
            svp_nh3, lh_nh3 = thermo.nh3_nh3h2osolution_saturation_vapor_pressure(
                T2, conc, Z
            )
            svp_h2o = svp_h2o * 1e5  # Convert to Pa
            svp_nh3 = svp_nh3 * 1e5
            if (svp_h2o / p2) < x1_h2o:
                # Condensation is occuring, use current concentration to compute change in mole fraction
                # For the solution, include both H2O and NH3 contributions
                dlnT_sol = dT / T2
                dlnP_sol = dp / p2
                n_nh3 = x1_nh3 / x_dry_air  # Molar mixing ratio wrt dry air
                individual_h2o = lh_h2o / (Z * spc.R * T2) * dlnT_sol - dlnP_sol
                correction_sol = n_nh3 * (
                    lh_nh3 / (Z * spc.R * T2) * dlnT_sol - dlnP_sol
                )
                dxw = x1_h2o * (individual_h2o + correction_sol)
                dxw = max(dxw, (svp_h2o / p2) - x1_h2o)
                dxa = conc / (1 - conc) * dxw
                c2_nh3h2o = conc
                solution_flag = True
            else:
                # Condensation isn't occuring after all
                dxw = 0.0
                dxa = 0.0
                lh_h2o = 0.0
                lh_nh3 = 0.0
                c2_nh3h2o = c1_nh3h2o
                solution_flag = False

            x2_h2o = x1_h2o + dxw
            x2_nh3 = x1_nh3 + dxa

    return (
        x2_h2o,
        x2_nh3,
        lh_h2o,
        lh_nh3,
        c2_nh3h2o,
        solution_flag,
    )


@njit(Tuple((f64, f64))(f64, f64, f64, f64, f64), cache=True)
def h2s_nh3h2osolution_difference(temperature, pressure, dxh2o, dxnh3, xh2s):
    """Computes how much H2S dissolves into an ammonia solution cloud
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
    X = (dxnh3 * thermo.NH3_MOLAR_MASS) / (
        dxnh3 * thermo.NH3_MOLAR_MASS + dxh2o * thermo.H2O_MOLAR_MASS
    )  # Mass fraction in g/g
    DENSOL = 0.9991 + X * (
        -0.4336
        + X * (0.3303 + X * (0.2833 + X * (-1.9716 + X * (2.1396 - X * 0.7294))))
    )  # Solution density in g/cm3

    # Calculate the partial pressure of H2S in the atmosphere in mmHg
    PPH2S = xh2s * pressure

    # Find the concentration of the solution cloud in moles of NH3 / liter
    VOLSOL = (
        -(dxh2o * thermo.H2O_MOLAR_MASS + dxnh3 * thermo.NH3_MOLAR_MASS)
        / DENSOL
        * litre_of_cm3
    )
    CONSOL = -dxnh3 / VOLSOL

    # Iterative method to calculate the concentration of H2S in solution
    FUNTMP = np.exp(22.221 - 5388.0 / temperature)
    FUNNH3 = CONSOL**1.8953
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


@njit(
    Tuple((arr1dc, arr1dc, arr2dc, f64, f64, f64, f64))(
        f64, f64, f64, f64, arr1d, f64, f64, f64
    )
)
def gas_cloud_condense(p1, T1, p2, T2, x_prev, c_prev_nh3h2o, x_dry_air, Z):
    """Compute gas condensation and cloud formation for a single pressure step.

    Uses a priority system:
      1. NH3-H2O solution (claims H2O and NH3)
      2. H2S dissolution into solution (if solution formed)
      3. NH4SH reaction (claims NH3 and H2S)
      4. Condensation for any unclaimed gas

    Parameters
    ----------
    p1 : float
        Pressure at previous level in Pa.
    T1 : float
        Temperature at previous level in K.
    p2 : float
        Pressure at current level in Pa.
    T2 : float
        Temperature at current level in K.
    x_prev : ndarray
        Mole fractions at the current level, shape (N_GASES,).
    c_prev_nh3h2o : float
        NH3 concentration in the NH3-H2O solution from previous level.
    x_dry_air : float
        Mole fraction of dry air (H2 + He) at the current level.
    Z : float
        Compressibility factor (dimensionless).

    Returns
    -------
    x_next : ndarray
        Updated mole fractions, shape (N_GASES,).
    lh : ndarray
        Latent heats released per gas in J/mol, shape (N_GASES,).
    aerosol : ndarray
        Aerosol densities in g/m^3, shape (N_GASES, N_PHASES).
    a_nh4sh : float
        NH4SH aerosol density in g/m^3.
    a_solution : float
        NH3-H2O solution aerosol density in g/m^3.
    c_nh3h2o : float
        Updated NH3 solution concentration.
    lh_nh4sh : float
        Latent heat from NH4SH reaction in J/mol.
    """
    dT = T2 - T1
    dp = p2 - p1

    x_next = x_prev.copy()
    lh = np.zeros(thermo.N_GASES)
    aerosol = np.zeros((thermo.N_GASES, thermo.N_PHASES))
    claimed = np.zeros(thermo.N_GASES, dtype=np.bool_)

    # Reaction products
    a_nh4sh = 0.0
    a_solution = 0.0
    c_nh3h2o = c_prev_nh3h2o
    lh_nh4sh = 0.0

    threshold = 1e-12

    # --- Priority 1: NH3-H2O solution (when both H2O and NH3 present) ---
    if x_prev[thermo.H2O_ID] > 0 and x_prev[thermo.NH3_ID] > 0:
        (
            x2_h2o,
            x2_nh3,
            lh_h2o,
            lh_nh3,
            c2_nh3h2o,
            solution_flag,
        ) = update_nh3_h2o_solution(
            p1,
            T1,
            p2,
            T2,
            x_prev[thermo.H2O_ID],
            x_prev[thermo.NH3_ID],
            c_prev_nh3h2o,
            x_dry_air,
            Z,
        )

        uc_h2o = update_cloud(
            p1,
            p2,
            T2,
            x_prev[thermo.H2O_ID],
            x2_h2o,
            thermo.MOLAR_MASS_ARRAY[thermo.H2O_ID],
            Z,
        )
        uc_nh3 = update_cloud(
            p1,
            p2,
            T2,
            x_prev[thermo.NH3_ID],
            x2_nh3,
            thermo.MOLAR_MASS_ARRAY[thermo.NH3_ID],
            Z,
        )

        if solution_flag:
            a_solution = uc_h2o + uc_nh3
            c_nh3h2o = c2_nh3h2o
            x_next[thermo.H2O_ID] = x2_h2o
            x_next[thermo.NH3_ID] = x2_nh3
            lh[thermo.H2O_ID] = lh_h2o
            lh[thermo.NH3_ID] = lh_nh3
            claimed[thermo.H2O_ID] = True
            claimed[thermo.NH3_ID] = True

        else:
            # Pass through to regular condensation, ignoring outputs of update_nh3_h2o_solution
            c_nh3h2o = c2_nh3h2o

        # H2S dissolution into NH3-H2O solution (if solution formed and H2S present)
        if x_prev[thermo.H2S_ID] > 0 and solution_flag:
            dxh2o = x2_h2o - x_prev[thermo.H2O_ID]
            dxnh3 = x2_nh3 - x_prev[thermo.NH3_ID]
            dxh2s, ch2s = h2s_nh3h2osolution_difference(
                T2, p2, dxh2o, dxnh3, x_prev[thermo.H2S_ID]
            )
            x_next[thermo.H2S_ID] = x_prev[thermo.H2S_ID] + dxh2s
        # After dissolution, NH4SH can still proceed with remaining NH3 and H2S

    elif x_prev[thermo.H2O_ID] > 0 and x_prev[thermo.NH3_ID] == 0:
        c_nh3h2o = 0.0

    # --- Priority 2: NH4SH (NH3 + H2S reaction) ---
    if x_prev[thermo.NH3_ID] > 0 and x_prev[thermo.H2S_ID] > 0:
        # Use current x_next values (may have been modified by solution/dissolution above)
        x_nh3_for_nh4sh = x_next[thermo.NH3_ID]
        x_h2s_for_nh4sh = x_next[thermo.H2S_ID]
        if x_nh3_for_nh4sh > 0 and x_h2s_for_nh4sh > 0:
            x2_h2s, x2_nh3, lh_nh4sh = update_nh4sh(
                p1, T1, p2, T2, x_h2s_for_nh4sh, x_nh3_for_nh4sh
            )
            a_nh4sh = update_cloud(
                p1, p2, T2, x_h2s_for_nh4sh, x2_h2s, thermo.NH4SH_MOLAR_MASS, Z
            )
            x_next[thermo.H2S_ID] = x2_h2s
            x_next[thermo.NH3_ID] = x2_nh3
            claimed[thermo.NH3_ID] = True
            claimed[thermo.H2S_ID] = True

    # --- Priority 3: Coupled condensation ---
    # First pass: determine which unclaimed gases are condensing, collect their L and phase
    condensing = np.zeros(thermo.N_GASES, dtype=np.bool_)
    lh_vals = np.zeros(thermo.N_GASES)
    phases = np.zeros(thermo.N_GASES, dtype=np.int64)
    svp_x = np.zeros(thermo.N_GASES)  # saturation mole fraction

    for g in range(thermo.N_GASES):
        if x_prev[g] > 0 and not claimed[g]:
            svp, lh_g, phase_g = svp_dispatch(g, T2, Z)
            svp_pa = svp * 1e5
            if (svp_pa / p2) < x_prev[g]:
                condensing[g] = True
                lh_vals[g] = lh_g
                phases[g] = phase_g
                svp_x[g] = svp_pa / p2

    # Second pass: compute the coupled correction sum
    # Σ_j n_j * (L_j/(RT²)*dT - dP/P) for all condensing gases j
    dlnT = dT / T2
    dlnP = dp / p2
    correction_sum = 0.0
    for g in range(thermo.N_GASES):
        if condensing[g]:
            n_g = x_prev[g] / x_dry_air
            correction_sum += n_g * (lh_vals[g] / (Z * spc.R * T2) * dlnT - dlnP)

    # Third pass: apply coupled dx to each condensing gas
    for g in range(thermo.N_GASES):
        if condensing[g]:
            # d ln X_i = L_i/(RT)*d ln T - d ln P + correction_sum
            individual_term = lh_vals[g] / (Z * spc.R * T2) * dlnT - dlnP
            n_g = x_prev[g] / x_dry_air
            this_correction = (
                correction_sum - n_g * individual_term
            )  # Correction doesn't apply to self
            dx = x_prev[g] * (individual_term + this_correction)
            dx = max(dx, svp_x[g] - x_prev[g])  # don't overshoot saturation
            x_next[g] = x_prev[g] + dx
            lh[g] = lh_vals[g]
            aerosol[g, phases[g]] = update_cloud(
                p1, p2, T2, x_prev[g], x_next[g], thermo.MOLAR_MASS_ARRAY[g], Z
            )

    # --- Truncate small values ---
    for g in range(thermo.N_GASES):
        if x_next[g] < threshold:
            x_next[g] = 0.0
            lh[g] = 0.0
            aerosol[g, thermo.SOLID] = 0.0
            aerosol[g, thermo.LIQUID] = 0.0
    # Also zero reaction products if reactants are gone
    if x_next[thermo.NH3_ID] < threshold or x_next[thermo.H2S_ID] < threshold:
        lh_nh4sh = 0.0
        a_nh4sh = 0.0

    return x_next, lh, aerosol, a_nh4sh, a_solution, c_nh3h2o, lh_nh4sh


@njit(
    Tuple((arr1dc, arr2dc, arr3dc, arr1dc, arr1dc, arr1dc, arr1dc))(
        arr1dc, arr1dc, arr1dc, arr1dc, f64, f64, b1, b1, b1, i64, f64, f64, f64, i64
    )
)
def run_eccm_core(
    pressure_grid,
    temperature_grid,
    deep_x,
    rh,
    bulk_h2,
    bulk_he,
    latent_heat_update,
    force_temperature,
    use_compressibility,
    cloud_model,
    f_sed,
    T_eff,
    planet_gravity,
    h2_type,
):
    """Run the equilibrium cloud condensation model integration loop.

    Integrates the atmosphere level-by-level from the deepest pressure to
    the top, computing condensation, cloud formation, and optionally
    applying wet adiabatic lapse rate corrections and sedimentation.

    Parameters
    ----------
    pressure_grid : ndarray
        Pressure levels in Pa, shape (n_levels,), sorted descending.
    temperature_grid : ndarray
        Initial temperature profile in K, shape (n_levels,). Modified in-place.
    deep_x : ndarray
        Deep (well-mixed) mole fractions, shape (N_GASES,).
    rh : ndarray
        Relative humidity factors, shape (N_GASES,).
    bulk_h2 : float
        Bulk H2 mole fraction.
    bulk_he : float
        Bulk He mole fraction.
    latent_heat_update : bool
        If True, recompute temperature with wet adiabatic lapse rate.
    force_temperature : bool
        If True, use the input temperature_grid as-is (no lapse rate update).
    use_compressibility : bool
        If True, apply non-ideal EOS correction (Z factor).
    cloud_model : int
        0 = equilibrium, 1 = sedimentation (Ackerman & Marley).
    f_sed : float
        Sedimentation efficiency parameter (cloud_model=1 only).
    T_eff : float
        Planet effective temperature in K (cloud_model=1 only).
    planet_gravity : float
        Surface gravity in m/s^2 (cloud_model=1 only).

    Returns
    -------
    temperature_grid : ndarray
        Updated temperature profile in K.
    x : ndarray
        Gas mole fraction profiles, shape (N_GASES, n_levels).
    aerosol : ndarray
        Aerosol densities in g/m^3, shape (N_GASES, N_PHASES, n_levels).
    a_nh4sh : ndarray
        NH4SH aerosol density profile in g/m^3.
    a_solution : ndarray
        NH3-H2O solution density profile in g/m^3.
    c_nh3h2o : ndarray
        NH3 solution concentration profile.
    z_profile : ndarray
        Compressibility factor profile (dimensionless).
    """
    n_levels = len(pressure_grid)

    x = np.zeros((thermo.N_GASES, n_levels))
    x[:, 0] = deep_x
    x_dry_air = bulk_h2 + bulk_he

    aerosol = np.zeros((thermo.N_GASES, thermo.N_PHASES, n_levels))
    a_nh4sh = np.zeros(n_levels)
    a_solution = np.zeros(n_levels)
    c_nh3h2o = np.zeros(n_levels)
    z_profile = np.ones(n_levels)

    # Sedimentation model cloud mixing ratios (mol/mol)
    q_c_sed = np.zeros((thermo.N_GASES, n_levels))
    q_c_sed_nh4sh = np.zeros(n_levels)

    if deep_x[thermo.NH3_ID] > 0 and deep_x[thermo.H2O_ID] > 0:
        c_nh3h2o[0] = deep_x[thermo.NH3_ID] / (
            deep_x[thermo.NH3_ID] + deep_x[thermo.H2O_ID]
        )

    for i in range(n_levels - 1):
        p1 = pressure_grid[i]
        p2 = pressure_grid[i + 1]
        pmean = 0.5 * (p1 + p2)
        dp = p2 - p1
        T1 = temperature_grid[i]

        # Compute compressibility factor at this level
        if use_compressibility:
            x_ch4_i = x[thermo.CH4_ID, i]
            x_h2o_i = x[thermo.H2O_ID, i]
            Z = compute_Z(pmean, T1, bulk_h2, bulk_he, x_ch4_i, x_h2o_i)
        else:
            Z = 1.0
        z_profile[i] = Z

        if use_compressibility:
            cp = compute_Cp(pmean, T1, Z, bulk_h2, bulk_he, x_ch4_i, x_h2o_i, h2_type)
        else:
            cp = (
                bulk_h2 * thermo.h2_normal_molar_heat_capacity(T1)
                + bulk_he * thermo.HE_MOLAR_HEAT_CAPACITY
            )
            tx = bulk_h2 + bulk_he
            for g in range(thermo.N_GASES):
                tx += x[g, i]
                cp += x[g, i] * thermo.MOLAR_HEAT_CAPACITY_ARRAY[g]
            cp = cp / tx  # In case mole fraction doesn't sum to 1.0

        if force_temperature:
            T2 = temperature_grid[i + 1]
        else:
            # Update temperature, dry adiabatic lapse rate (Z correction)
            dTdp = Z * spc.R * T1 / cp / pmean
            T2 = T1 + dTdp * dp

        # Gas/cloud condensation
        x_next, lh_arr, aer, a_nh4sh_i, a_sol_i, c_i, lh_nh4sh = gas_cloud_condense(
            p1, T1, p2, T2, x[:, i], c_nh3h2o[i], x_dry_air, Z
        )

        # Recalculate temperature with wet adiabatic lapse rate
        if latent_heat_update and not force_temperature:
            x_cur = x[:, i]
            # Numerator: R*T * (1 + Σ L*n/(RT)) → R*T + Σ L*n
            # Denominator: R*P * (cp/R + (Σ L²n/(RT)² + (Σ Ln/(RT))² / (1+Σn))
            sum_x = 0.0
            sum_Ln = 0.0
            sum_L2n = 0.0
            dTdp_numerator = Z * spc.R * T1

            for g in range(thermo.N_GASES):
                sum_x += x_cur[g]
                n_g = x_cur[g] / x_dry_air  # mixing ratio
                dTdp_numerator += lh_arr[g] * n_g
                Ln_over_RT = lh_arr[g] * n_g / (Z * spc.R * T1)
                sum_Ln += Ln_over_RT
                sum_L2n += lh_arr[g] ** 2 * n_g / (Z * spc.R * T1) ** 2

            sum_n = sum_x / x_dry_air
            dTdp_denominator = pmean * cp + pmean * Z * spc.R * (
                sum_L2n + sum_Ln**2
            ) / (1.0 + sum_n)

            # NH4SH contribution (uses its own equilibrium constant approach)
            if x_cur[thermo.NH3_ID] > 0 and x_cur[thermo.H2S_ID] > 0:
                dTdp_numerator += (
                    2
                    * lh_nh4sh
                    * x_cur[thermo.NH3_ID]
                    * x_cur[thermo.H2S_ID]
                    / (x_cur[thermo.NH3_ID] + x_cur[thermo.H2S_ID])
                )
                dTdp_denominator += (
                    pmean
                    * lh_nh4sh
                    * x_cur[thermo.NH3_ID]
                    * x_cur[thermo.H2S_ID]
                    / (x_cur[thermo.NH3_ID] + x_cur[thermo.H2S_ID])
                    * 10834
                    / T1**2
                )

            dTdp = dTdp_numerator / dTdp_denominator
            T2 = T1 + dTdp * dp

            # Recompute condensation with updated temperature
            x_next, lh_arr, aer, a_nh4sh_i, a_sol_i, c_i, lh_nh4sh = gas_cloud_condense(
                p1, T1, p2, T2, x[:, i], c_nh3h2o[i], x_dry_air, Z
            )

        # Sedimentation cloud model: override aerosol computation
        if cloud_model == 1:
            # Compute needed quantities for sedimentation model
            M_mix = (
                bulk_h2 * thermo.H2_MOLAR_MASS + bulk_he * thermo.HE_MOLAR_MASS
            )  # g/mol
            H = spc.R * T2 / (M_mix * 1e-3 * planet_gravity)  # scale height in m
            if use_compressibility:
                cp_sed = compute_Cp(
                    p2, T2, Z, bulk_h2, bulk_he, x_ch4_i, x_h2o_i, h2_type
                )
            else:
                cp_sed = (
                    bulk_h2 * thermo.h2_normal_molar_heat_capacity(T2)
                    + bulk_he * thermo.HE_MOLAR_HEAT_CAPACITY
                )  # J/(mol*K)

            # Dry adiabatic lapse rate in K/m
            gamma_dry = Z * M_mix * 1e-3 * planet_gravity / cp_sed

            # Wet lapse rate: use the ratio dT/dz from the temperature step
            if abs(dp) > 0:
                actual_dTdz = (T2 - T1) / (-H * dp / pmean)  # K/m
            else:
                actual_dTdz = gamma_dry
            gamma_wet = max(actual_dTdz, gamma_dry * 0.1)

            # Layer thickness in m (positive, going upward)
            dz = H * abs(dp) / pmean

            # Detect solution condensation and reset for A&M recomputation
            solution_at_level = a_sol_i != 0.0
            if solution_at_level:
                a_sol_i = 0.0

            nh4sh_at_level = a_nh4sh_i != 0.0
            if nh4sh_at_level:
                # Use H2S as the vapor proxy (consumed 1:1 to form NH4SH)
                q_v_nh4sh = x_next[thermo.H2S_ID]
                q_v_prev_nh4sh = x[thermo.H2S_ID, i]
                q_c_prev_nh4sh = q_c_sed_nh4sh[i]

                # At cloud base (first NH4SH), use equilibrium excess
                if q_c_prev_nh4sh == 0.0:
                    dx_nh4sh = x_next[thermo.H2S_ID] - x[thermo.H2S_ID, i]
                    q_c_prev_nh4sh = -dx_nh4sh  # positive: H2S consumed

                q_c_nh4sh_new = compute_cloud_sediment(
                    T2,
                    p2,
                    T_eff,
                    H,
                    gamma_wet,
                    gamma_dry,
                    dz,
                    q_v_nh4sh,
                    q_v_prev_nh4sh,
                    q_c_prev_nh4sh,
                    f_sed,
                    M_mix,
                    cp_sed,
                )
                q_c_sed_nh4sh[i + 1] = q_c_nh4sh_new

                # Convert to NH4SH density (g/m³)
                a_nh4sh_i = (
                    abs(q_c_nh4sh_new)
                    * thermo.NH4SH_MOLAR_MASS
                    * p2**2
                    / (spc.R * T2 * abs(dp))
                )

            for g in range(thermo.N_GASES):
                # Check if condensation occurred for this gas
                x_prev_g = x[g, i]
                dx_g = x_next[g] - x_prev_g
                if dx_g < 0:  # condensation occurred (vapor decreased)
                    # Skip gases consumed by NH4SH reaction
                    if nh4sh_at_level:
                        if g == thermo.H2S_ID:
                            q_c_sed[g, i + 1] = 0.0
                            continue
                        if g == thermo.NH3_ID and not solution_at_level:
                            q_c_sed[g, i + 1] = 0.0
                            continue

                    # Previous level cloud mixing ratio
                    q_c_prev_g = q_c_sed[g, i]
                    # At cloud base (first condensation), use the equilibrium excess
                    if q_c_prev_g == 0.0:
                        q_c_prev_g = -dx_g  # amount condensed (positive)

                    q_c_new = compute_cloud_sediment(
                        T2,
                        p2,
                        T_eff,
                        H,
                        gamma_wet,
                        gamma_dry,
                        dz,
                        x_next[g],
                        x_prev_g,
                        q_c_prev_g,
                        f_sed,
                        M_mix,
                        cp_sed,
                    )
                    q_c_sed[g, i + 1] = q_c_new

                    # Convert per-layer condensate to density (g/m³)
                    aer_density = (
                        abs(q_c_new)
                        * thermo.MOLAR_MASS_ARRAY[g]
                        * p2**2
                        / (spc.R * T2 * abs(dp))
                    )

                    if solution_at_level and (g == thermo.H2O_ID or g == thermo.NH3_ID):
                        # Solution cloud: accumulate into combined solution density
                        a_sol_i += aer_density
                    else:
                        # Individual condensation: store in aerosol array
                        svp_val, lh_val, phase_val = svp_dispatch(g, T2, Z)
                        aerosol[g, phase_val, i + 1] = aer_density
                else:
                    q_c_sed[g, i + 1] = 0.0

        # Store results
        temperature_grid[i + 1] = T2
        x[:, i + 1] = x_next
        if cloud_model == 0:
            aerosol[:, :, i + 1] = aer
        a_nh4sh[i + 1] = a_nh4sh_i
        a_solution[i + 1] = a_sol_i
        c_nh3h2o[i + 1] = c_i

    # Set Z for last level
    if use_compressibility:
        z_profile[-1] = compute_Z(
            pressure_grid[-1],
            temperature_grid[-1],
            bulk_h2,
            bulk_he,
            x[thermo.CH4_ID, -1],
            x[thermo.H2O_ID, -1],
        )

    # Apply relative humidity adjustment
    for g in range(thermo.N_GASES):
        condensed = np.zeros(n_levels, dtype=np.bool_)
        for k in range(n_levels):
            if aerosol[g, thermo.SOLID, k] + aerosol[g, thermo.LIQUID, k] != 0:
                condensed[k] = True
        # Also check solution for H2O
        if g == thermo.H2O_ID:
            for k in range(n_levels):
                if a_solution[k] != 0:
                    condensed[k] = True
        if condensed.any():
            idx = -1
            for k in range(n_levels):
                if condensed[k]:
                    idx = k
                    break
            if idx >= 0:
                x[g, idx:] = x[g, idx:] * rh[g]

    return temperature_grid, x, aerosol, a_nh4sh, a_solution, c_nh3h2o, z_profile
