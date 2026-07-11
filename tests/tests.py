"""
ECCM Tests
Run with: pytest tests.py -v
"""

import numpy as np
import pytest


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def jupiter_pressure_grid():
    return np.logspace(np.log10(100e5), np.log10(0.1e5), 300)


@pytest.fixture
def jupiter_reference():
    ref_p = np.array([0.1e5, 1e5, 5e5])
    ref_t = np.array([110.0, 165.0, 260.0])
    return ref_p, ref_t


@pytest.fixture
def jupiter_gases():
    from eccm_mrtm.eccm import GasInput

    return [
        GasInput("H2O", deep=1e-3),
        GasInput("NH3", deep=1.5e-4),
        GasInput("H2S", deep=3e-5),
        GasInput("CH4", deep=5e-4),
        GasInput("PH3", deep=5e-7),
    ]


@pytest.fixture
def simple_gases():
    from eccm_mrtm.eccm import GasInput

    return [GasInput("NH3", deep=1.5e-4), GasInput("CH4", deep=5e-4)]


# ==============================================================================
# Compressibility Factor
# ==============================================================================


class TestCompressibility:
    def test_z_disabled(self, jupiter_pressure_grid, jupiter_reference, simple_gases):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        result = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            use_compressibility=False,
        )
        assert np.allclose(result["compressibility_profile"], 1.0)

    def test_z_greater_than_one_at_high_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Z

        Z = compute_Z(100e5, 300.0, 0.864, 0.136, 0.0, 0.0)
        assert Z > 1.01

    def test_z_near_one_at_low_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Z

        Z = compute_Z(1e5, 300.0, 0.864, 0.136, 0.0, 0.0)
        assert Z < 1.001

    def test_z_approaches_one_at_zero_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Z

        Z = compute_Z(100.0, 300.0, 0.864, 0.136, 0.0, 0.0)
        assert abs(Z - 1.0) < 1e-5

    def test_z_monotonically_increases_with_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Z

        pressures = [1e5, 5e5, 10e5, 50e5, 100e5]
        Z_values = [compute_Z(P, 300.0, 0.864, 0.136, 0.0, 0.0) for P in pressures]
        for i in range(len(Z_values) - 1):
            assert Z_values[i + 1] >= Z_values[i]

    def test_compressibility_warms_deep_atmosphere(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        r_ideal = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            use_compressibility=False,
        )
        r_real = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            use_compressibility=True,
        )
        assert r_real["temperature"][0] > r_ideal["temperature"][0]


# ==============================================================================
# Condensation
# ==============================================================================


class TestCondense:
    def test_coupled_correction_in_gas_cloud_condense(self):
        """When multiple gases condense simultaneously, the coupled correction
        Σ_j n_j*(L_j/(RT)*dlnT - dlnP) enhances condensation for all gases."""
        from eccm_mrtm.eccm.core import gas_cloud_condense
        from eccm_mrtm.eccm.thermo import H2O_ID, NH3_ID, N_GASES
        import numpy as np

        # Set up conditions where both H2O and NH3 condense independently
        # (below freezing, so solution doesn't form)
        x_prev = np.zeros(N_GASES)
        x_prev[H2O_ID] = 0.05  # 5% H2O — significant!
        x_prev[NH3_ID] = 0.01  # 1% NH3
        # Cold enough that both condense as solid (below NH3-H2O solution freezing)
        T2 = 150.0
        T1 = 153.0
        p1 = 3.1e5
        p2 = 3.0e5
        Z = 1.0
        x_dry_air = 0.864 + 0.136  # bulk_h2 + bulk_he
        result_multi = gas_cloud_condense(p1, T1, p2, T2, x_prev, 0.0, x_dry_air, Z)
        x_next_multi = result_multi[0]

        # Compare: run with only H2O (no other condensing gas)
        x_prev_h2o_only = np.zeros(N_GASES)
        x_prev_h2o_only[H2O_ID] = 0.05
        result_single = gas_cloud_condense(
            p1, T1, p2, T2, x_prev_h2o_only, 0.0, x_dry_air, Z
        )
        x_next_single = result_single[0]

        # With NH3 also condensing, the correction sum is larger,
        # so H2O dx should be enhanced (more negative)
        dx_multi = x_next_multi[H2O_ID] - x_prev[H2O_ID]
        dx_single = x_next_single[H2O_ID] - x_prev_h2o_only[H2O_ID]
        if dx_multi < 0 and dx_single < 0:
            assert abs(dx_multi) > abs(dx_single), (
                "Coupled correction should enhance H2O condensation when NH3 also condenses"
            )


# ==============================================================================
# Section 4: Sedimentation Cloud Model
# ==============================================================================


class TestSedimentationCloud:
    def test_equilibrium_produces_clouds(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        result = run_eccm(
            jupiter_pressure_grid, ref_p, ref_t, 24.79, simple_gases, 0.864, 0.136
        )
        assert abs(result["aerosol_densities"]["NH3"]["solid"]).max() > 0

    def test_sediment_thinner_than_equilibrium(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        r_eq = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="equilibrium",
        )
        r_sed = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="sediment",
            cloud_kwargs={"f_sed": 3.0, "T_eff": 124.0},
        )
        eq_max = abs(r_eq["aerosol_densities"]["NH3"]["solid"]).max()
        sed_max = abs(r_sed["aerosol_densities"]["NH3"]["solid"]).max()
        assert sed_max < eq_max

    @pytest.mark.parametrize("f_sed", [0.5, 1.0, 3.0, 5.0])
    def test_higher_fsed_gives_thinner_clouds(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases, f_sed
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        r_low = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="sediment",
            cloud_kwargs={"f_sed": f_sed, "T_eff": 124.0},
        )
        r_high = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="sediment",
            cloud_kwargs={"f_sed": f_sed + 1.0, "T_eff": 124.0},
        )
        low_max = abs(r_low["aerosol_densities"]["NH3"]["solid"]).max()
        high_max = abs(r_high["aerosol_densities"]["NH3"]["solid"]).max()
        assert high_max <= low_max

    def test_sediment_does_not_change_gas_profiles(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        r_eq = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="equilibrium",
        )
        r_sed = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            cloud_model="sediment",
            cloud_kwargs={"f_sed": 2.0, "T_eff": 124.0},
        )
        assert np.allclose(r_eq["gas_profiles"]["NH3"], r_sed["gas_profiles"]["NH3"])
        assert np.allclose(r_eq["temperature"], r_sed["temperature"])


# ==============================================================================
# Gas Module System
# ==============================================================================


class TestGasModuleSystem:
    def test_gas_index_mapping(self):
        from eccm_mrtm.eccm.thermo import (
            GAS_INDEX,
            H2O_ID,
            NH3_ID,
            H2S_ID,
            CH4_ID,
            PH3_ID,
        )

        assert GAS_INDEX["H2O"] == H2O_ID == 0
        assert GAS_INDEX["NH3"] == NH3_ID == 1
        assert GAS_INDEX["H2S"] == H2S_ID == 2
        assert GAS_INDEX["CH4"] == CH4_ID == 3
        assert GAS_INDEX["PH3"] == PH3_ID == 4

    def test_gasinput_and_result_structure(self):
        from eccm_mrtm.eccm import run_eccm, GasInput

        P = np.logspace(np.log10(50e5), np.log10(0.5e5), 100)
        ref_p = np.array([0.5e5, 2e5])
        ref_t = np.array([130.0, 200.0])
        gases = [GasInput("NH3", deep=1.5e-4), GasInput("CH4", deep=5e-4)]
        result = run_eccm(P, ref_p, ref_t, 24.79, gases, 0.864, 0.136)
        assert isinstance(result, dict)
        assert "NH3" in result["gas_profiles"]
        assert "CH4" in result["gas_profiles"]
        assert "solid" in result["aerosol_densities"]["NH3"]
        assert "liquid" in result["aerosol_densities"]["NH3"]
        assert result["pressure"].shape == result["temperature"].shape
        assert result["compressibility_profile"].shape == result["pressure"].shape

    def test_empty_gas_list(self):
        from eccm_mrtm.eccm import run_eccm

        P = np.logspace(np.log10(50e5), np.log10(0.5e5), 50)
        ref_p = np.array([0.5e5, 2e5])
        ref_t = np.array([130.0, 200.0])
        result = run_eccm(P, ref_p, ref_t, 24.79, [], 0.864, 0.136)
        assert not np.isnan(result["temperature"]).any()


# ==============================================================================
# Full Run (Jupiter-like)
# ==============================================================================


class TestFullRun:
    def test_jupiter_runs_clean(
        self, jupiter_pressure_grid, jupiter_reference, jupiter_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        result = run_eccm(
            jupiter_pressure_grid, ref_p, ref_t, 24.79, jupiter_gases, 0.864, 0.136
        )
        assert not np.isnan(result["temperature"]).any()
        assert result["temperature"].min() > 50
        assert result["temperature"].max() < 800
        assert (result["aerosol_densities"]["NH4SH"]["solid"] != 0).any()
        assert (result["aerosol_densities"]["H2O_NH3_SOLUTION"]["liquid"] != 0).any()

    def test_jupiter_with_compressibility(
        self, jupiter_pressure_grid, jupiter_reference, jupiter_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        result = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            jupiter_gases,
            0.864,
            0.136,
            use_compressibility=True,
        )
        assert not np.isnan(result["temperature"]).any()
        Z_max = result["compressibility_profile"].max()
        assert 1.02 < Z_max < 1.10

    def test_jupiter_with_latent_heat(
        self, jupiter_pressure_grid, jupiter_reference, jupiter_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        result = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            jupiter_gases,
            0.864,
            0.136,
            latent_heat_update=True,
        )
        assert not np.isnan(result["temperature"]).any()

    def test_relative_humidity(self):
        from eccm_mrtm.eccm import run_eccm, GasInput

        P = np.logspace(np.log10(50e5), np.log10(0.01e5), 300)
        ref_p = np.array([0.1e5, 1e5, 5e5])
        ref_t = np.array([110.0, 165.0, 260.0])
        r_full = run_eccm(
            P, ref_p, ref_t, 24.79, [GasInput("H2S", deep=5e-5, rh=1.0)], 0.864, 0.136
        )
        r_half = run_eccm(
            P, ref_p, ref_t, 24.79, [GasInput("H2S", deep=5e-5, rh=0.5)], 0.864, 0.136
        )
        cond_idx = np.argmax(r_full["gas_profiles"]["H2S"] < 5e-5)
        if cond_idx > 0:
            check = cond_idx + 5
            if check < len(P) and r_full["gas_profiles"]["H2S"][check] > 0:
                ratio = (
                    r_half["gas_profiles"]["H2S"][check]
                    / r_full["gas_profiles"]["H2S"][check]
                )
                assert 0.45 < ratio < 0.55


# ==============================================================================
# Thermodynamic Properties
# ==============================================================================


class TestThermo:
    def test_solution_h2o_matches_pure_at_c0(self):
        from eccm_mrtm.eccm.thermo import (
            h2o_liquid_saturation_vapor_pressure,
            h2o_nh3h2osolution_saturation_vapor_pressure,
        )

        for T in [250.0, 300.0, 350.0]:
            pure_svp, pure_lh = h2o_liquid_saturation_vapor_pressure(T, 1.0)
            sol_svp, sol_lh = h2o_nh3h2osolution_saturation_vapor_pressure(T, 0.0, 1.0)
            assert np.isclose(pure_svp, sol_svp, rtol=1e-10)
            assert np.isclose(pure_lh, sol_lh, rtol=1e-10)

    def test_solution_nh3_matches_pure_at_c1(self):
        from eccm_mrtm.eccm.thermo import (
            nh3_liquid_saturation_vapor_pressure,
            nh3_nh3h2osolution_saturation_vapor_pressure,
        )

        for T in [200.0, 250.0, 300.0]:
            pure_svp, pure_lh = nh3_liquid_saturation_vapor_pressure(T, 1.0)
            sol_svp, sol_lh = nh3_nh3h2osolution_saturation_vapor_pressure(T, 1.0, 1.0)
            assert np.isclose(pure_svp, sol_svp, rtol=1e-10)
            assert np.isclose(pure_lh, sol_lh, rtol=1e-10)

    def test_solution_h2o_decreases_with_concentration(self):
        from eccm_mrtm.eccm.thermo import h2o_nh3h2osolution_saturation_vapor_pressure

        T = 300.0
        svps = [h2o_nh3h2osolution_saturation_vapor_pressure(T, c, 1.0)[0]
                for c in [0.0, 0.2, 0.4, 0.6, 0.8]]
        for i in range(len(svps) - 1):
            assert svps[i] > svps[i + 1]

    def test_solution_nh3_decreases_with_dilution(self):
        from eccm_mrtm.eccm.thermo import nh3_nh3h2osolution_saturation_vapor_pressure

        T = 250.0
        svps = [nh3_nh3h2osolution_saturation_vapor_pressure(T, c, 1.0)[0]
                for c in [1.0, 0.8, 0.6, 0.4, 0.2]]
        for i in range(len(svps) - 1):
            assert svps[i] > svps[i + 1]


# ==============================================================================
# Heat Capacity
# ==============================================================================


class TestCp:
    def test_cp_greater_than_ideal_at_high_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Cp
        from eccm_mrtm.eccm.thermo import h2_normal_molar_heat_capacity, HE_MOLAR_HEAT_CAPACITY

        T = 300.0
        x_h2, x_he = 0.864, 0.136
        cp_real = compute_Cp(100e5, T, x_h2, x_he, 0.0, 0.0, 0)
        cp_ideal = x_h2 * h2_normal_molar_heat_capacity(T) + x_he * HE_MOLAR_HEAT_CAPACITY
        assert cp_real > cp_ideal

    def test_cp_approaches_ideal_at_low_pressure(self):
        from eccm_mrtm.eccm.eos import compute_Cp
        from eccm_mrtm.eccm.thermo import h2_normal_molar_heat_capacity, HE_MOLAR_HEAT_CAPACITY

        T = 300.0
        x_h2, x_he = 0.864, 0.136
        cp_real = compute_Cp(1e5, T, x_h2, x_he, 0.0, 0.0, 0)
        cp_ideal = x_h2 * h2_normal_molar_heat_capacity(T) + x_he * HE_MOLAR_HEAT_CAPACITY
        assert abs(cp_real - cp_ideal) / cp_ideal < 0.01


# ==============================================================================
# Condensation (additional tests)
# ==============================================================================


class TestCondenseAdditional:
    def test_single_gas_condensation(self):
        from eccm_mrtm.eccm import run_eccm, GasInput

        P = np.logspace(np.log10(50e5), np.log10(0.1e5), 200)
        ref_p = np.array([0.1e5, 1e5, 5e5])
        ref_t = np.array([110.0, 165.0, 260.0])
        result = run_eccm(P, ref_p, ref_t, 24.79, [GasInput("NH3", deep=1.5e-4)], 0.864, 0.136)
        nh3_aer = result["aerosol_densities"]["NH3"]
        total_cloud = abs(nh3_aer["solid"]).max() + abs(nh3_aer["liquid"]).max()
        assert total_cloud > 0

    def test_nh4sh_reaction(self):
        from eccm_mrtm.eccm.core import update_nh4sh

        p1 = 5.1e5
        T1 = 240.0
        p2 = 5.0e5
        T2 = 239.0
        x_h2s = 3e-5
        x_nh3 = 1.5e-4
        x2_h2s, x2_nh3, lh = update_nh4sh(p1, T1, p2, T2, x_h2s, x_nh3)
        # If reaction occurs, both should decrease by same amount
        if lh > 0:
            dx_h2s = x2_h2s - x_h2s
            dx_nh3 = x2_nh3 - x_nh3
            assert np.isclose(dx_h2s, dx_nh3)
            assert dx_h2s < 0


# ==============================================================================
# Utilities
# ==============================================================================


class TestUtilities:
    def test_solar_concentration_keys(self):
        from eccm_mrtm.eccm import solar_concentration

        result = solar_concentration("asplund2009")
        expected_keys = {"H2", "He", "C", "N", "O", "P", "S"}
        assert set(result.keys()) == expected_keys

    def test_solar_concentration_h2_is_one(self):
        from eccm_mrtm.eccm import solar_concentration

        for model in ["asplund2009", "asplund2021", "lodders2025"]:
            result = solar_concentration(model)
            assert np.isclose(result["H2"], 1.0)

    def test_solar_concentration_he_reasonable(self):
        from eccm_mrtm.eccm import solar_concentration

        result = solar_concentration("asplund2009")
        assert 0.15 < result["He"] < 0.25


# ==============================================================================
# Full Run (additional)
# ==============================================================================


class TestFullRunAdditional:
    def test_latent_heat_warms_cloud_zone(
        self, jupiter_pressure_grid, jupiter_reference, simple_gases
    ):
        from eccm_mrtm.eccm import run_eccm

        ref_p, ref_t = jupiter_reference
        r_dry = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            latent_heat_update=False,
        )
        r_wet = run_eccm(
            jupiter_pressure_grid,
            ref_p,
            ref_t,
            24.79,
            simple_gases,
            0.864,
            0.136,
            latent_heat_update=True,
        )
        # In the condensation zone, latent heat release should make T warmer
        # Find where NH3 condenses (aerosol > 0)
        nh3_cloud = abs(r_dry["aerosol_densities"]["NH3"]["solid"])
        cloud_mask = nh3_cloud > 0
        if cloud_mask.any():
            # Temperature should be warmer (or equal) with latent heat at cloud levels
            T_diff = r_wet["temperature"][cloud_mask] - r_dry["temperature"][cloud_mask]
            assert T_diff.mean() >= 0


# ==============================================================================
# Static Compilation
# ==============================================================================


class TestNumbaCompilation:
    def test_all_core_functions_compiled(self):
        from eccm_mrtm.eccm import core

        funcs = [
            "svp_dispatch",
            "update_single_gas",
            "update_cloud",
            "update_nh4sh",
            "fn_nh3",
            "fn_h2o",
            "fn_both",
            "update_nh3_h2o_solution",
            "h2s_nh3h2osolution_difference",
            "gas_cloud_condense",
            "run_eccm_core",
            "compute_cloud_sediment",
        ]
        for name in funcs:
            f = getattr(core, name)
            assert len(f.signatures) > 0, f"{name} has no compiled signatures"
