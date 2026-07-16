import numpy as np
from numba import njit
from numba.core.types import float64, int64, Array, Tuple


CONVERGENCE_PRECISION = 1e-6
# Type aliases
f64 = float64
i64 = int64
arr1d = Array(float64, 1, "A")  # 1D float64 array (any layout)
arr1d_ro = Array(
    float64, 1, "C", readonly=True
)  # 1D float64 array (readonly, C-contiguous)

# ==============================================================================
# Compressibility Factor (Z) Module
# Full Helmholtz free energy-based equation of state for H2-He-CH4-H2O mixtures
# Based on IAPWS-04 (H2), mBWR (He), GERG-2008 (CH4), IAPWS-95 (H2O)
# Note: Much of this is taken from Bryan Karpowicz's LRTM repository, https://github.com/karpob/lrtm
# and Karpowicz, B. M., & Steffes, P. G. (2013). https://doi.org/10.1016/j.icarus.2012.11.026
# ==============================================================================

R_UNIVERSAL = 8.314462618  # J/(mol*K)

# ==============================================================================
# GAS PARAMETERS
# ==============================================================================

# --- Hydrogen (normal H2) - IAPWS-04, 14 terms ---
H2_TC = 33.145  # K
H2_RHOC = 31.263  # kg/m3 (= 15.508 mol/L * 2.01594 g/mol)
H2_M = 2.01594e-3  # kg/mol
H2_RHOC_MOL = H2_RHOC / (H2_M * 1000)  # mol/L -> mol/m3: 15508 mol/m3

H2_N_I = np.array(
    [
        -6.93643,
        0.01,
        2.1101,
        4.52059,
        0.732564,
        -1.34086,
        0.130985,
        -0.777414,
        0.351944,
        -0.0211716,
        0.0226312,
        0.032187,
        -0.0231752,
        0.0557346,
    ]
)
H2_T_I = np.array(
    [
        0.6844,
        1.0,
        0.989,
        0.489,
        0.803,
        1.1444,
        1.409,
        1.754,
        1.311,
        4.187,
        5.646,
        0.791,
        7.249,
        2.986,
    ]
)
H2_D_I = np.array(
    [1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0]
)
H2_P_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
H2_PHI_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.685, 0.489, 0.103, 2.506, 1.607]
)
H2_BETA_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.171, 0.2245, 0.1304, 0.2785, 0.3967]
)
H2_GAMMA_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.7164,
        1.3444,
        1.4517,
        0.7204,
        1.5445,
    ]
)
H2_D_GAUSS_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.506, 0.156, 1.736, 0.67, 1.662]
)
H2_N_POW_NOEXP = 7
H2_N_POW_WEXP = 2
H2_N_GAUSS = 5
H2_N_CRIT = 0

# --- Helium - mBWR converted, 80 terms ---
HE_TC = 5.1953  # K
HE_RHOC = 69.641  # kg/m3 (= 17.399 mol/L * 4.0026 g/mol)
HE_M = 4.0026e-3  # kg/mol

HE_N_I = np.array(
    [
        9.5403824222421316e-02,
        9.8560755822836858e-05,
        -1.8841600266314820e-01,
        2.0143163965979105e-06,
        -8.4611385367388032e-05,
        -1.6137775659564457e-04,
        6.4447060292073915e-08,
        -3.4618307429427857e-06,
        3.2029019468536735e-05,
        3.8318860272147829e-06,
        1.8502065494236440e-07,
        -8.9572466296080700e-07,
        -3.2015284694056307e-04,
        1.9511512147139807e-03,
        1.9809530350459699e-04,
        -4.2105833106133463e-08,
        -1.5046915371820254e-02,
        1.1574487205410517e00,
        -2.8758406999199089e00,
        7.5429412526775652e-01,
        -2.3717709285388028e-01,
        2.6474346332976149e-02,
        -2.1380000968551547e-03,
        -1.6498337532806573e-01,
        7.6413223711691991e-01,
        3.1297894783730495e-01,
        7.3954941795512796e-05,
        1.4930962085205435e-02,
        -1.0755875976072855e-03,
        1.0973233079587626e-01,
        8.2315428494380573e-04,
        -3.0935483755047039e-01,
        -3.2029019468536735e-05,
        -7.3954941795512796e-05,
        -3.6977470897756398e-05,
        -1.2325823632585467e-05,
        -3.0814559081463667e-06,
        1.8841600266314820e-01,
        1.8841600266314820e-01,
        -5.3381699114227895e-06,
        9.4208001331574101e-02,
        7.8506667776311757e-03,
        -7.3954941795512796e-05,
        1.5701333555262350e-03,
        -1.6014509734268368e-05,
        1.6137775659564457e-04,
        8.0688878297822286e-05,
        2.6896292765940764e-05,
        6.7240731914851910e-06,
        -3.2029019468536735e-05,
        1.3448146382970381e-06,
        3.1402667110524703e-02,
        1.6137775659564457e-04,
        8.4611385367388032e-05,
        -1.6426792637139475e-05,
        8.9572466296080700e-07,
        -1.8502065494236440e-07,
        -3.8318860272147829e-06,
        -3.8318860272147829e-06,
        4.2105833106133463e-08,
        4.2105833106133463e-08,
        3.4618307429427857e-06,
        3.4618307429427857e-06,
        1.7309153714713929e-06,
        -6.4447060292073915e-08,
        -6.4447060292073915e-08,
        -3.2223530146036957e-08,
        8.4611385367388032e-05,
        -1.3345424778556974e-06,
        4.2305692683694016e-05,
        1.4101897561231339e-05,
        -2.0143163965979105e-06,
        -2.0143163965979105e-06,
        -1.0071581982989553e-06,
        -3.3571939943298507e-07,
        -9.8560755822836858e-05,
        -9.8560755822836858e-05,
        -4.9280377911418429e-05,
        -4.1066981592848688e-06,
        -2.6690849557113949e-07,
    ]
)
HE_T_I = np.array(
    [
        0.0,
        3.0,
        3.0,
        5.0,
        3.0,
        4.0,
        4.0,
        3.0,
        5.0,
        3.0,
        4.0,
        3.0,
        3.0,
        3.0,
        2.0,
        5.0,
        3.0,
        0.5,
        1.0,
        2.0,
        3.0,
        0.0,
        2.0,
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        0.0,
        1.0,
        1.0,
        2.0,
        5.0,
        4.0,
        4.0,
        4.0,
        4.0,
        3.0,
        3.0,
        5.0,
        3.0,
        3.0,
        4.0,
        3.0,
        5.0,
        4.0,
        4.0,
        4.0,
        4.0,
        5.0,
        4.0,
        3.0,
        4.0,
        3.0,
        3.0,
        3.0,
        4.0,
        3.0,
        3.0,
        5.0,
        5.0,
        3.0,
        3.0,
        3.0,
        4.0,
        4.0,
        4.0,
        3.0,
        5.0,
        3.0,
        3.0,
        5.0,
        5.0,
        5.0,
        5.0,
        3.0,
        3.0,
        3.0,
        3.0,
        5.0,
    ]
)
HE_D_I = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        7.0,
        7.0,
        0.0,
        5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        6.0,
        2.0,
        2.0,
        2.0,
        0.0,
        5.0,
        3.0,
        3.0,
        4.0,
        3.0,
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        0.0,
        2.0,
        6.0,
        4.0,
        8.0,
        0.0,
        10.0,
        4.0,
        0.0,
        4.0,
        6.0,
        8.0,
        2.0,
        10.0,
        6.0,
        2.0,
        2.0,
        6.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        2.0,
        0.0,
        2.0,
        4.0,
        0.0,
        2.0,
        4.0,
        0.0,
        8.0,
        4.0,
        6.0,
        0.0,
        2.0,
        4.0,
        6.0,
        0.0,
        2.0,
        4.0,
        8.0,
        10.0,
    ]
)
HE_P_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
)
HE_N_POW_NOEXP = 32
HE_N_POW_WEXP = 48
HE_N_GAUSS = 0
HE_N_CRIT = 0

# --- Methane (CH4) - 40 terms ---
CH4_TC = 190.564  # K
CH4_RHOC = 162.66  # kg/m3 (= 10.139128 mol/L * 16.0428 g/mol)
CH4_M = 16.0428e-3  # kg/mol

CH4_N_I = np.array(
    [
        0.43679010280e-01,
        0.67092361990,
        -0.17655778590e01,
        0.85823302410,
        -0.12065130520e01,
        0.51204672200,
        -0.40000107910e-03,
        -0.12478424230e-01,
        0.31002697010e-01,
        0.17547485220e-02,
        -0.31719216050e-05,
        -0.22403468400e-05,
        0.29470561560e-06,
        0.18304879090,
        0.15118836790,
        -0.42893638770,
        0.68940024460e-01,
        -0.14083139960e-01,
        -0.30630548300e-01,
        -0.29699067080e-01,
        -0.19320408310e-01,
        -0.11057399590,
        0.99525489950e-01,
        0.85484378250e-02,
        -0.61505556620e-01,
        -0.42917924230e-01,
        -0.18132072900e-01,
        0.34459047600e-01,
        -0.23859194500e-02,
        -0.11590949390e-01,
        0.66416936020e-01,
        -0.23715495900e-01,
        -0.39616249050e-01,
        -0.13872920440e-01,
        0.33894895990e-01,
        -0.29273787530e-02,
        0.93247999460e-04,
        -0.62871715180e01,
        0.12710694670e02,
        -0.64239534660e01,
    ]
)
CH4_T_I = np.array(
    [
        -0.5,
        0.5,
        1.0,
        0.5,
        1.0,
        1.5,
        4.5,
        0.0,
        1.0,
        3.0,
        1.0,
        3.0,
        3.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        2.0,
        2.0,
        5.0,
        5.0,
        5.0,
        2.0,
        4.0,
        12.0,
        8.0,
        10.0,
        10.0,
        10.0,
        14.0,
        12.0,
        18.0,
        22.0,
        18.0,
        14.0,
        2.0,
        0.0,
        1.0,
        2.0,
    ]
)
CH4_D_I = np.array(
    [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        4.0,
        4.0,
        8.0,
        9.0,
        10.0,
        1.0,
        1.0,
        1.0,
        2.0,
        4.0,
        5.0,
        6.0,
        1.0,
        2.0,
        3.0,
        4.0,
        4.0,
        3.0,
        5.0,
        5.0,
        8.0,
        2.0,
        3.0,
        4.0,
        4.0,
        4.0,
        5.0,
        6.0,
        2.0,
        0.0,
        0.0,
        0.0,
    ]
)
CH4_P_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
    ],
    dtype=np.float64,
)
CH4_PHI_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        20.0,
        40.0,
        40.0,
        40.0,
    ],
    dtype=np.float64,
)
CH4_BETA_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        200,
        250,
        250,
        250,
    ],
    dtype=np.float64,
)
CH4_GAMMA_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1.07,
        1.11,
        1.11,
        1.11,
    ],
    dtype=np.float64,
)
CH4_D_GAUSS_I = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ],
    dtype=np.float64,
)
CH4_N_POW_NOEXP = 13
CH4_N_POW_WEXP = 23
CH4_N_GAUSS = 4
CH4_N_CRIT = 0

# --- Water (H2O) - IAPWS-95, 56 terms ---
H2O_TC = 647.096  # K
H2O_RHOC = 322.0  # kg/m3
H2O_M = 18.015268e-3  # kg/mol

H2O_N_I = np.array(
    [
        0.12533547935523e-1,
        0.78957634722828e1,
        -0.87803203303561e1,
        0.31802509345418,
        -0.26145533859358,
        -0.78199751687981e-2,
        0.88089493102134e-2,
        -0.66856572307965,
        0.20433810950965,
        -0.66212605039687e-4,
        -0.19232721156002,
        -0.25709043003438,
        0.16074868486251,
        -0.40092828925807e-1,
        0.39343422603254e-6,
        -0.75941377088144e-5,
        0.56250979351888e-3,
        -0.15608652257135e-4,
        0.11537996422951e-8,
        0.36582165144204e-6,
        -0.13251180074668e-11,
        -0.62639586912454e-9,
        -0.10793600908932,
        0.17611491008752e-1,
        0.22132295167546,
        -0.40247669763528,
        0.58083399985759,
        0.49969146990806e-2,
        -0.31358700712549e-1,
        -0.74315929710341,
        0.47807329915480,
        0.20527940895948e-1,
        -0.13636435110343,
        0.14180634400617e-1,
        0.83326504880713e-2,
        -0.29052336009585e-1,
        0.38615085574206e-1,
        -0.20393486513704e-1,
        -0.16554050063734e-2,
        0.19955571979541e-2,
        0.15870308324157e-3,
        -0.16388568342530e-4,
        0.43613615723811e-1,
        0.34994005463765e-1,
        -0.76788197844621e-1,
        0.22446277332006e-1,
        -0.62689710414685e-4,
        -0.55711118565645e-9,
        -0.19905718354408,
        0.31777497330738,
        -0.11841182425981,
        -0.31306260323435e2,
        0.31546140237781e2,
        -0.25213154341695e4,
        -0.14874640856724,
        0.31806110878444,
    ]
)
H2O_T_I = np.array(
    [
        -0.5,
        0.875,
        1.0,
        0.5,
        0.75,
        0.375,
        1.0,
        4.0,
        6.0,
        12.0,
        1.0,
        5.0,
        4.0,
        2.0,
        13.0,
        9.0,
        3.0,
        4.0,
        11.0,
        4.0,
        13.0,
        1.0,
        7.0,
        1.0,
        9.0,
        10.0,
        10.0,
        3.0,
        7.0,
        10.0,
        10.0,
        6.0,
        10.0,
        10.0,
        1.0,
        2.0,
        3.0,
        4.0,
        8.0,
        6.0,
        9.0,
        8.0,
        16.0,
        22.0,
        23.0,
        23.0,
        10.0,
        50.0,
        44.0,
        46.0,
        50.0,
        0.0,
        1.0,
        4.0,
    ]
)
H2O_D_I = np.array(
    [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        4.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        4.0,
        4.0,
        5.0,
        7.0,
        9.0,
        10.0,
        11.0,
        13.0,
        15.0,
        1.0,
        2.0,
        2.0,
        2.0,
        3.0,
        4.0,
        4.0,
        4.0,
        5.0,
        6.0,
        6.0,
        7.0,
        9.0,
        9.0,
        9.0,
        9.0,
        9.0,
        10.0,
        10.0,
        12.0,
        3.0,
        4.0,
        4.0,
        5.0,
        14.0,
        3.0,
        6.0,
        6.0,
        6.0,
        3.0,
        3.0,
        3.0,
    ]
)
H2O_P_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
        4.0,
        6.0,
        6.0,
        6.0,
        6.0,
    ]
)
# Gaussian terms for H2O (indices 51-53)
H2O_PHI_I = np.zeros(54)
H2O_PHI_I[51] = 20.0
H2O_PHI_I[52] = 20.0
H2O_PHI_I[53] = 20.0
H2O_BETA_I = np.zeros(56)
H2O_BETA_I[51] = 150.0
H2O_BETA_I[52] = 150.0
H2O_BETA_I[53] = 250.0
H2O_BETA_I[54] = 0.3
H2O_BETA_I[55] = 0.3
H2O_GAMMA_I = np.zeros(54)
H2O_GAMMA_I[51] = 1.21
H2O_GAMMA_I[52] = 1.21
H2O_GAMMA_I[53] = 1.25
H2O_D_GAUSS_I = np.zeros(54)
H2O_D_GAUSS_I[51] = 1.0
H2O_D_GAUSS_I[52] = 1.0
H2O_D_GAUSS_I[53] = 1.0
# Critical terms for H2O (indices 54-55)
H2O_RES_A = np.zeros(56)
H2O_RES_A[54] = 0.32
H2O_RES_A[55] = 0.32
H2O_RES_B = np.zeros(56)
H2O_RES_B[54] = 0.2
H2O_RES_B[55] = 0.2
H2O_RES_C = np.zeros(56)
H2O_RES_C[54] = 28.0
H2O_RES_C[55] = 32.0
H2O_RES_D = np.zeros(56)
H2O_RES_D[54] = 700.0
H2O_RES_D[55] = 800.0
H2O_RES_a = np.zeros(56)
H2O_RES_a[54] = 3.5
H2O_RES_a[55] = 3.5
H2O_RES_b = np.zeros(56)
H2O_RES_b[54] = 0.85
H2O_RES_b[55] = 0.95
H2O_N_POW_NOEXP = 7
H2O_N_POW_WEXP = 44
H2O_N_GAUSS = 3
H2O_N_CRIT = 2

# --- Binary mixture: H2-CH4 (GERG-2008) ---
H2CH4_N_I = np.array(
    [-0.25157134971934, -0.62203841111983e-2, 0.88850315184396e-1, -0.35592212573239e-1]
)
H2CH4_T_I = np.array([2.0, -1.0, 1.75, 1.4])
H2CH4_D_I = np.array([1.0, 3.0, 3.0, 4.0])
H2CH4_P_I = np.array([0.0, 0.0, 0.0, 0.0])
H2CH4_BETAT = 1.0
H2CH4_BETAV = 1.0
H2CH4_GAMMAT = 1.352643115
H2CH4_GAMMAV = 1.018702573
H2CH4_N_POW_NOEXP = 4
H2CH4_N_POW_WEXP = 0

# --- Binary mixture: H2-H2O (fitted) ---
H2H2O_N_I = np.array([0.84730166e-01, 0.120304163e-01, 4.85353759, -9.45732780])
H2H2O_T_I = np.array([29.6892622, 5.66963126, -0.472763978, 5.68600592])
H2H2O_D_I = np.array([1.01325950, 0.875427966, 2.25904893, 1.73721803])
H2H2O_P_I = np.array([0.157106640, -0.123114242, 1.07298418, 0.751254725])
H2H2O_BETAT = -68.4724158
H2H2O_BETAV = 2.76510561
H2H2O_GAMMAT = -172.902015
H2H2O_GAMMAV = 3.36805346
H2H2O_N_POW_NOEXP = 0
H2H2O_N_POW_WEXP = 4


# ==============================================================================
# CORE HELMHOLTZ DERIVATIVE FUNCTION
# ==============================================================================


@njit(
    f64(
        f64,
        f64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        i64,
        i64,
        i64,
        i64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
    ),
    cache=True,
)
def d_alpha_d_delta_pure(
    tau,
    delta,
    N_i,
    t_i,
    d_i,
    p_i,
    phi_i,
    beta_i,
    gamma_i,
    D_i,
    n_pow_noexp,
    n_pow_wexp,
    n_gauss,
    n_crit,
    RES_a,
    RES_b,
    RES_B,
    RES_C,
    RES_D_crit,
    RES_A,
):
    """Compute d(alpha_residual)/d(delta) for a pure substance Helmholtz EOS.

    Evaluates the derivative of the dimensionless residual Helmholtz energy
    with respect to reduced density for a single pure component or binary
    interaction departure function. The compressibility factor is then:
    Z = 1 + delta * d(alpha_r)/d(delta).

    Based on IAPWS-04 (H2), mBWR (He), GERG-2008 (CH4), IAPWS-95 (H2O).

    Parameters
    ----------
    tau : float
        Inverse reduced temperature, Tc_mix / T (dimensionless).
    delta : float
        Reduced density, rho / rho_c_mix (dimensionless).
    N_i : ndarray
        Amplitude coefficients for each EOS term.
    t_i : ndarray
        Temperature exponents for each term.
    d_i : ndarray
        Density exponents for each term.
    p_i : ndarray
        Exponential density exponents (power-with-exp terms).
    phi_i : ndarray
        Gaussian term distance parameters.
    beta_i : ndarray
        Gaussian term shape parameters.
    gamma_i : ndarray
        Gaussian term center parameters for tau.
    D_i : ndarray
        Gaussian term center parameters for delta.
    n_pow_noexp : int
        Number of power terms without exponential damping.
    n_pow_wexp : int
        Number of power terms with exponential damping.
    n_gauss : int
        Number of Gaussian terms.
    n_crit : int
        Number of non-analytic critical-region terms.
    RES_a, RES_b, RES_B, RES_C, RES_D_crit, RES_A : ndarray
        Critical term parameters (used only for H2O).

    Returns
    -------
    float
        d(alpha_r)/d(delta) (dimensionless).
    """
    result = 0.0

    # Term 1: Power terms without exponential (p_i == 0)
    for i in range(n_pow_noexp):
        if d_i[i] != 0.0:
            result += N_i[i] * d_i[i] * delta ** (d_i[i] - 1.0) * tau ** t_i[i]

    # Term 2: Power terms with exponential (p_i != 0)
    idx_start = n_pow_noexp
    for i in range(n_pow_wexp):
        k = idx_start + i
        p = p_i[k]
        d = d_i[k]
        if d != 0.0:
            exp_term = np.exp(-(delta**p))
            result += (
                N_i[k]
                * exp_term
                * delta ** (d - 1.0)
                * tau ** t_i[k]
                * (d - p * delta**p)
            )
        else:
            # d_i == 0 means this is a pure exponential term (no delta power)
            p = p_i[k]
            exp_term = np.exp(-(delta**p))
            result += N_i[k] * exp_term * tau ** t_i[k] * (-p * delta ** (p - 1.0))

    # Term 3: Gaussian terms
    idx_start = n_pow_noexp + n_pow_wexp
    for i in range(n_gauss):
        k = idx_start + i
        phi = phi_i[k]
        beta = beta_i[k]
        gam = gamma_i[k]
        D_val = D_i[k]
        d = d_i[k]
        gauss_exp = np.exp(-phi * (delta - D_val) ** 2 - beta * (tau - gam) ** 2)
        result += (
            N_i[k]
            * delta**d
            * tau ** t_i[k]
            * gauss_exp
            * (d / delta - 2.0 * phi * (delta - D_val))
        )

    # Term 4: Critical terms (H2O only)
    idx_start = n_pow_noexp + n_pow_wexp + n_gauss
    for i in range(n_crit):
        k = idx_start + i
        a_val = RES_a[k]
        b_val = RES_b[k]
        beta_val = beta_i[k]
        B_val = RES_B[k]
        C_val = RES_C[k]
        D_val = RES_D_crit[k]
        A_val = RES_A[k]

        # Intermediate quantities
        dm1_sq = (delta - 1.0) ** 2
        theta = (1.0 - tau) + A_val * dm1_sq ** (0.5 / beta_val)
        Delta = theta**2 + B_val * dm1_sq**a_val
        Psi = np.exp(-C_val * dm1_sq - D_val * (tau - 1.0) ** 2)

        # Derivatives
        dPsi_ddelta = -2.0 * C_val * (delta - 1.0) * Psi
        dDelta_ddelta = (delta - 1.0) * (
            A_val * theta * 2.0 / beta_val * dm1_sq ** (0.5 / beta_val - 1.0)
            + 2.0 * B_val * a_val * dm1_sq ** (a_val - 1.0)
        )

        if Delta > 0.0:
            Delta_bm1 = Delta ** (b_val - 1.0)
            Delta_b = Delta**b_val
            dDeltab_ddelta = b_val * Delta_bm1 * dDelta_ddelta
        else:
            Delta_b = 0.0
            dDeltab_ddelta = 0.0

        result += N_i[k] * (
            Delta_b * (Psi + delta * dPsi_ddelta) + dDeltab_ddelta * delta * Psi
        )

    return result


@njit(
    f64(
        f64,
        f64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        i64,
        i64,
        i64,
        i64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
    ),
    cache=True,
)
def d_alpha_d_tau_d_tau_pure(
    tau,
    delta,
    N_i,
    t_i,
    d_i,
    p_i,
    phi_i,
    beta_i,
    gamma_i,
    D_i,
    n_pow_noexp,
    n_pow_wexp,
    n_gauss,
    n_crit,
    RES_a,
    RES_b,
    RES_B,
    RES_C,
    RES_D_crit,
    RES_A,
):
    result = 0.0

    for i in range(n_pow_noexp):
        result += (
            N_i[i] * t_i[i] * (t_i[i] - 1.0) * delta ** d_i[i] * tau ** (t_i[i] - 2.0)
        )

    idx_start = n_pow_noexp
    for i in range(n_pow_wexp):
        k = idx_start + i
        result += (
            N_i[k]
            * t_i[k]
            * (t_i[k] - 1.0)
            * delta ** d_i[k]
            * tau ** (t_i[k] - 2.0)
            * np.exp(-(delta ** p_i[k]))
        )

    idx_start = n_pow_noexp + n_pow_wexp
    for i in range(n_gauss):
        k = idx_start + i
        phi = phi_i[k]
        beta = beta_i[k]
        gam = gamma_i[k]
        D_val = D_i[k]
        d = d_i[k]
        t = t_i[k]
        gauss_exp = np.exp(-phi * (delta - D_val) ** 2 - beta * (tau - gam) ** 2)
        p2 = (t / tau - 2.0 * beta * (tau - gam)) ** 2 - t / tau**2 - 2.0 * beta
        result += N_i[k] * delta**d * tau**t * gauss_exp * p2

    idx_start = n_pow_noexp + n_pow_wexp + n_gauss
    for i in range(n_crit):
        k = idx_start + i
        a_val = RES_a[k]
        b_val = RES_b[k]
        beta_val = beta_i[k]
        B_val = RES_B[k]
        C_val = RES_C[k]
        D_val = RES_D_crit[k]
        A_val = RES_A[k]

        dm1_sq = (delta - 1.0) ** 2
        theta = (1.0 - tau) + A_val * dm1_sq ** (0.5 / beta_val)
        Delta = theta**2 + B_val * dm1_sq**a_val
        Psi = np.exp(-C_val * dm1_sq - D_val * (tau - 1.0) ** 2)

        dDeltab_dtau = (
            -2.0 * theta * b_val * Delta ** (b_val - 1.0) if Delta > 0.0 else 0.0
        )
        dPsi_dtau = -2.0 * D_val * (tau - 1.0) * Psi
        d2Deltab_dtau = (
            (
                2.0 * b_val * Delta ** (b_val - 1.0)
                + 4.0 * theta**2 * b_val * (b_val - 1.0) * Delta ** (b_val - 2.0)
            )
            if Delta > 0.0
            else 0.0
        )
        d2Psi_dtau = (2.0 * D_val * (tau - 1.0) ** 2 - 1.0) * 2.0 * D_val * Psi

        Delta_b = Delta**b_val if Delta > 0.0 else 0.0
        result += (
            N_i[k]
            * delta
            * (
                d2Deltab_dtau * Psi
                + 2.0 * dDeltab_dtau * dPsi_dtau
                + Delta_b * d2Psi_dtau
            )
        )

    return result


@njit(
    f64(
        f64,
        f64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        i64,
        i64,
        i64,
        i64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
    ),
    cache=True,
)
def d_alpha_d_delta_d_tau_pure(
    tau,
    delta,
    N_i,
    t_i,
    d_i,
    p_i,
    phi_i,
    beta_i,
    gamma_i,
    D_i,
    n_pow_noexp,
    n_pow_wexp,
    n_gauss,
    n_crit,
    RES_a,
    RES_b,
    RES_B,
    RES_C,
    RES_D_crit,
    RES_A,
):
    result = 0.0

    for i in range(n_pow_noexp):
        result += (
            N_i[i] * d_i[i] * t_i[i] * delta ** (d_i[i] - 1.0) * tau ** (t_i[i] - 1.0)
        )

    idx_start = n_pow_noexp
    for i in range(n_pow_wexp):
        k = idx_start + i
        p = p_i[k]
        d = d_i[k]
        t = t_i[k]
        exp_term = np.exp(-(delta**p))
        result += (
            N_i[k]
            * t
            * delta ** (d - 1.0)
            * tau ** (t - 1.0)
            * (d - p * delta**p)
            * exp_term
        )

    idx_start = n_pow_noexp + n_pow_wexp
    for i in range(n_gauss):
        k = idx_start + i
        phi = phi_i[k]
        beta = beta_i[k]
        gam = gamma_i[k]
        D_val = D_i[k]
        d = d_i[k]
        t = t_i[k]
        gauss_exp = np.exp(-phi * (delta - D_val) ** 2 - beta * (tau - gam) ** 2)
        f1 = t - 2.0 * beta * tau * (tau - gam)
        g1 = d - 2.0 * phi * delta * (delta - D_val)
        result += N_i[k] * f1 * tau ** (t - 1.0) * g1 * delta ** (d - 1.0) * gauss_exp

    idx_start = n_pow_noexp + n_pow_wexp + n_gauss
    for i in range(n_crit):
        k = idx_start + i
        a_val = RES_a[k]
        b_val = RES_b[k]
        beta_val = beta_i[k]
        B_val = RES_B[k]
        C_val = RES_C[k]
        D_val = RES_D_crit[k]
        A_val = RES_A[k]

        dm1 = delta - 1.0
        dm1_sq = dm1**2
        tm1 = tau - 1.0
        theta = (1.0 - tau) + A_val * dm1_sq ** (0.5 / beta_val)
        Delta = theta**2 + B_val * dm1_sq**a_val
        Psi = np.exp(-C_val * dm1_sq - D_val * tm1**2)

        dDeltab_dtau = (
            -2.0 * theta * b_val * Delta ** (b_val - 1.0) if Delta > 0.0 else 0.0
        )
        dPsi_dtau = -2.0 * D_val * tm1 * Psi
        dDelta_ddelta = dm1 * (
            A_val * theta * 2.0 / beta_val * dm1_sq ** (0.5 / beta_val - 1.0)
            + 2.0 * B_val * a_val * dm1_sq ** (a_val - 1.0)
        )
        dDeltab_ddelta = (
            b_val * Delta ** (b_val - 1.0) * dDelta_ddelta if Delta > 0.0 else 0.0
        )
        dPsi_ddelta = -2.0 * C_val * dm1 * Psi
        d2Psi_ddelta_dtau = 4.0 * C_val * D_val * dm1 * tm1 * Psi
        d2Deltab_ddelta_dtau = (
            (
                -A_val
                * b_val
                * 2.0
                / beta_val
                * Delta ** (b_val - 1.0)
                * dm1
                * dm1_sq ** (0.5 / beta_val - 1.0)
                - 2.0
                * theta
                * b_val
                * (b_val - 1.0)
                * Delta ** (b_val - 2.0)
                * dDelta_ddelta
            )
            if Delta > 0.0
            else 0.0
        )

        Delta_b = Delta**b_val if Delta > 0.0 else 0.0
        result += N_i[k] * (
            Delta_b * (dPsi_dtau + delta * d2Psi_ddelta_dtau)
            + delta * dDeltab_ddelta * dPsi_dtau
            + dDeltab_dtau * (Psi + delta * dPsi_ddelta)
            + d2Deltab_ddelta_dtau * delta * Psi
        )

    return result


@njit(
    f64(
        f64,
        f64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        i64,
        i64,
        i64,
        i64,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
        arr1d_ro,
    ),
    cache=True,
)
def d_alpha_d_delta_d_delta_pure(
    tau,
    delta,
    N_i,
    t_i,
    d_i,
    p_i,
    phi_i,
    beta_i,
    gamma_i,
    D_i,
    n_pow_noexp,
    n_pow_wexp,
    n_gauss,
    n_crit,
    RES_a,
    RES_b,
    RES_B,
    RES_C,
    RES_D_crit,
    RES_A,
):
    result = 0.0

    for i in range(n_pow_noexp):
        d = d_i[i]
        result += N_i[i] * d * (d - 1.0) * delta ** (d - 2.0) * tau ** t_i[i]

    idx_start = n_pow_noexp
    for i in range(n_pow_wexp):
        k = idx_start + i
        p = p_i[k]
        d = d_i[k]
        exp_term = np.exp(-(delta**p))
        factor1 = (d - p * delta**p) * (d - 1.0 - p * delta**p) - p**2 * delta**p
        result += N_i[k] * exp_term * delta ** (d - 2.0) * tau ** t_i[k] * factor1

    idx_start = n_pow_noexp + n_pow_wexp
    for i in range(n_gauss):
        k = idx_start + i
        phi = phi_i[k]
        beta = beta_i[k]
        gam = gamma_i[k]
        D_val = D_i[k]
        d = d_i[k]
        gauss_exp = np.exp(-phi * (delta - D_val) ** 2 - beta * (tau - gam) ** 2)
        pp2 = 2.0 * phi * delta**d
        pp3 = 4.0 * phi**2 * delta**d * (delta - D_val) ** 2
        pp4 = 4.0 * d * phi * delta ** (d - 1.0) * (delta - D_val)
        pp5 = d * (d - 1.0) * delta ** (d - 2.0)
        result += N_i[k] * tau ** t_i[k] * gauss_exp * (-pp2 + pp3 - pp4 + pp5)

    idx_start = n_pow_noexp + n_pow_wexp + n_gauss
    for i in range(n_crit):
        k = idx_start + i
        a_val = RES_a[k]
        b_val = RES_b[k]
        beta_val = beta_i[k]
        B_val = RES_B[k]
        C_val = RES_C[k]
        D_val = RES_D_crit[k]
        A_val = RES_A[k]

        dm1 = delta - 1.0
        dm1_sq = dm1**2
        theta = (1.0 - tau) + A_val * dm1_sq ** (0.5 / beta_val)
        Delta = theta**2 + B_val * dm1_sq**a_val
        Psi = np.exp(-C_val * dm1_sq - D_val * (tau - 1.0) ** 2)

        dDelta_ddelta = dm1 * (
            A_val * theta * 2.0 / beta_val * dm1_sq ** (0.5 / beta_val - 1.0)
            + 2.0 * B_val * a_val * dm1_sq ** (a_val - 1.0)
        )
        dPsi_ddelta = -2.0 * C_val * dm1 * Psi
        d2Psi_ddelta = 2.0 * C_val * Psi * (2.0 * C_val * dm1_sq - 1.0)

        ppp1 = 4.0 * B_val * a_val * (a_val - 1.0) * dm1_sq ** (a_val - 2.0)
        ppp2 = 2.0 * A_val**2 / beta_val**2 * (dm1_sq ** (0.5 / beta_val - 1.0)) ** 2
        ppp3 = (
            A_val
            * theta
            * (4.0 / beta_val)
            * (0.5 / beta_val - 1.0)
            * dm1_sq ** (0.5 / beta_val - 2.0)
        )
        d2Delta_ddelta = (
            dDelta_ddelta / dm1 + dm1_sq * (ppp1 + ppp2 + ppp3) if dm1 != 0.0 else 0.0
        )

        if Delta > 0.0:
            Delta_b = Delta**b_val
            dDeltab_ddelta = b_val * Delta ** (b_val - 1.0) * dDelta_ddelta
            d2Deltab_ddelta = b_val * (
                Delta ** (b_val - 1.0) * d2Delta_ddelta
                + (b_val - 1.0) * Delta ** (b_val - 2.0) * dDelta_ddelta**2
            )
        else:
            Delta_b = 0.0
            dDeltab_ddelta = 0.0
            d2Deltab_ddelta = 0.0

        result += N_i[k] * (
            Delta_b * (2.0 * dPsi_ddelta + delta * d2Psi_ddelta)
            + 2.0 * dDeltab_ddelta * (Psi + delta * dPsi_ddelta)
            + d2Deltab_ddelta * delta * Psi
        )

    return result


# ==============================================================================
# IDEAL HELMHOLTZ SECOND TAU-DERIVATIVE FUNCTIONS
# ==============================================================================


@njit(f64(f64, f64, arr1d_ro, arr1d_ro, f64), cache=True)
def ideal_alpha_dtau_dtau_cp(ni, Tc, vi, ui, T):
    """Compute Cp_ideal/R for Cp-type gases (H2, He, CH4).

    For a monatomic gas (He), vi/ui arrays are length 1 with value 0 —
    the loop contributes nothing and the result is just ni.
    """
    sumz = ni
    for i in range(len(vi)):
        if ui[i] > 0.0:
            x = ui[i] / T
            ea = np.exp(-x)
            d = (1.0 - ea) ** 2
            sumz += vi[i] * x * x * ea / d
    return sumz


@njit(f64(f64, arr1d_ro, arr1d_ro), cache=True)
def ideal_alpha_dtau_dtau_coef(tau, ideal_n, ideal_gamma):
    """Compute d²(alpha_ideal)/d(tau²) for Coef-type gases (H2O).

    Uses the IAPWS-95 ideal gas formulation with coefficient arrays.
    """
    result = -ideal_n[2] / (tau * tau)
    for i in range(3, len(ideal_n)):
        g = ideal_gamma[i]
        eg = np.exp(-g * tau)
        result -= ideal_n[i] * g * g * eg / (1.0 - eg) ** 2
    return result


# ==============================================================================
# H2 VARIANT PARAMETERS (Ortho and Para)
# ==============================================================================

# --- Ortho H2 - 14 terms ---
ORTHO_H2_TC = 32.22
ORTHO_H2_RHOC = 15.445 * 2.01594  # kg/m3
ORTHO_H2_NI = 2.5
ORTHO_H2_VI = np.array([2.54151, -2.3661, 1.00365, 1.22447])
ORTHO_H2_UI = np.array([856.0, 1444.0, 2194.0, 6968.0])
ORTHO_H2_N_I = np.array(
    [
        -6.83148,
        0.01,
        2.11505,
        4.38353,
        0.211292,
        -1.00939,
        0.142086,
        -0.87696,
        0.804927,
        -0.710775,
        0.0639688,
        0.0710858,
        -0.087654,
        0.647088,
    ]
)
ORTHO_H2_T_I = np.array(
    [
        0.7333,
        1.0,
        1.1372,
        0.5136,
        0.5638,
        1.6248,
        1.829,
        2.404,
        2.105,
        4.1,
        7.658,
        1.259,
        7.589,
        3.946,
    ]
)
ORTHO_H2_D_I = np.array(
    [1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0]
)
ORTHO_H2_P_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
ORTHO_H2_PHI_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.169, -0.894, -0.04, -2.072, -1.306]
)
ORTHO_H2_BETA_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4555,
        -0.4046,
        -0.0869,
        -0.4415,
        -0.5743,
    ]
)
ORTHO_H2_GAMMA_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5444, 0.6627, 0.763, 0.6587, 1.4327]
)
ORTHO_H2_D_GAUSS_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6366,
        0.3876,
        0.9437,
        0.3976,
        0.9626,
    ]
)

# --- Para H2 - 14 terms ---
PARA_H2_TC = 32.938
PARA_H2_RHOC = 15.538 * 2.01594  # kg/m3
PARA_H2_NI = 2.5
PARA_H2_VI = np.array(
    [4.30256, 13.0289, -47.7365, 50.0013, -18.6261, 0.993973, 0.536078]
)
PARA_H2_UI = np.array([499.0, 826.5, 970.8, 1166.2, 1341.4, 5395.0, 10185.0])
PARA_H2_N_I = np.array(
    [
        -7.33375,
        0.01,
        2.60375,
        4.66279,
        0.682390,
        -1.47078,
        0.135801,
        -1.05327,
        0.328239,
        -0.0577833,
        0.0449743,
        0.0703464,
        -0.0401766,
        0.119510,
    ]
)
PARA_H2_T_I = np.array(
    [
        0.6855,
        1.0,
        1.0,
        0.489,
        0.774,
        1.133,
        1.386,
        1.619,
        1.162,
        3.96,
        5.276,
        0.99,
        6.791,
        3.19,
    ]
)
PARA_H2_D_I = np.array(
    [1.0, 4.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0]
)
PARA_H2_P_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
PARA_H2_PHI_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.7437,
        -0.5516,
        -0.0634,
        -2.1341,
        -1.777,
    ]
)
PARA_H2_BETA_I = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.194,
        -0.2019,
        -0.0301,
        -0.2383,
        -0.3253,
    ]
)
PARA_H2_GAMMA_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8048, 1.5248, 0.6648, 0.6832, 1.493]
)
PARA_H2_D_GAUSS_I = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5487, 0.1785, 1.28, 0.6319, 1.7104]
)

# Normal H2 ideal gas parameters (for convenience alongside the existing residual params)
H2_NI = 2.5
H2_VI = np.array([1.616, -0.4117, -0.792, 0.758, 1.217])
H2_UI = np.array([531.0, 751.0, 1989.0, 2484.0, 6859.0])

# He ideal gas parameters
HE_NI = 2.5
HE_VI = np.zeros(1)
HE_UI = np.zeros(1)

# CH4 ideal gas parameters
CH4_NI = 4.0016
CH4_VI = np.array([0.84490000e-02, 4.6942000, 3.4865000, 1.6572000, 1.4115000])
CH4_UI = np.array([648.0, 1957.0, 3895.0, 5705.0, 15080.0])

# H2O ideal gas parameters (Coef type)
H2O_IDEAL_N = np.array(
    [
        -8.32044648201,
        6.6832105268,
        3.00632,
        0.012436,
        0.97315,
        1.27950,
        0.96956,
        0.24873,
    ]
)
H2O_IDEAL_GAMMA = np.array(
    [0.0, 0.0, 0.0, 1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105]
)


# ==============================================================================
# MIXTURE REDUCED VARIABLE SCALING (GERG quadratic mixing rules)
# ==============================================================================


@njit(Tuple((f64, f64))(f64, f64, f64, f64), cache=True)
def mixture_reducing_parameters(x_h2, x_he, x_ch4, x_h2o):
    """Compute mixture pseudo-critical parameters using GERG quadratic mixing rules.

    Determines the pseudo-critical temperature and molar density for an
    H2-He-CH4-H2O mixture. Binary interaction parameters are used for
    H2-CH4 and H2-H2O pairs; others default to Lorentz-Berthelot rules.

    Parameters
    ----------
    x_h2 : float
        Mole fraction of H2.
    x_he : float
        Mole fraction of He.
    x_ch4 : float
        Mole fraction of CH4.
    x_h2o : float
        Mole fraction of H2O.

    Returns
    -------
    Tc_mix : float
        Pseudo-critical temperature of the mixture in K.
    rhoc_mix_molL : float
        Pseudo-critical molar density of the mixture in mol/L.
    """
    # Critical molar densities (mol/m3)
    # Let's use mol/L for the mixing rule (matches the reference code)
    rhoc_h2_molL = 15.508  # mol/L
    rhoc_he_molL = 17.399  # mol/L
    rhoc_ch4_molL = 10.139128  # mol/L
    rhoc_h2o_molL = 322.0 / 18.015268  # mol/L = 17.874

    Tc = np.array([H2_TC, HE_TC, CH4_TC, H2O_TC])
    rhoc = np.array([rhoc_h2_molL, rhoc_he_molL, rhoc_ch4_molL, rhoc_h2o_molL])
    x = np.array([x_h2, x_he, x_ch4, x_h2o])

    # BetaT, GammaT matrices (symmetric, only off-diagonal matters)
    # Available pairs: H2-CH4 (0,2), H2-H2O (0,3)
    # Missing pairs default to BetaT=1, GammaT=1 (no interaction)

    # Compute Tc_mix
    Tc_mix = 0.0
    for i in range(4):
        Tc_mix += x[i] ** 2 * Tc[i]

    # Binary contributions to Tc_mix
    # H2-He (no correlations, use Beta=Gamma=1)
    Tc_mix += 2.0 * x_h2 * x_he * 1.0 * 1.0 * np.sqrt(Tc[0] * Tc[1])
    # H2-CH4
    Tc_mix += 2.0 * x_h2 * x_ch4 * (H2CH4_GAMMAT / H2CH4_BETAT) * np.sqrt(Tc[0] * Tc[2])
    # H2-H2O
    Tc_mix += 2.0 * x_h2 * x_h2o * (H2H2O_GAMMAT / H2H2O_BETAT) * np.sqrt(Tc[0] * Tc[3])
    # He-CH4, He-H2O, CH4-H2O (no correlations)
    Tc_mix += 2.0 * x_he * x_ch4 * 1.0 * 1.0 * np.sqrt(Tc[1] * Tc[2])
    Tc_mix += 2.0 * x_he * x_h2o * 1.0 * 1.0 * np.sqrt(Tc[1] * Tc[3])
    Tc_mix += 2.0 * x_ch4 * x_h2o * 1.0 * 1.0 * np.sqrt(Tc[2] * Tc[3])

    # Compute 1/rho_c_mix
    inv_rhoc_mix = 0.0
    for i in range(4):
        if x[i] > 0:
            inv_rhoc_mix += x[i] ** 2 / rhoc[i]

    # Binary contributions
    def rhoc_cross(rhoc_i, rhoc_j):
        return (
            1.0 / (rhoc_i ** (1.0 / 3.0)) + 1.0 / (rhoc_j ** (1.0 / 3.0))
        ) ** 3 / 8.0

    # H2-He
    inv_rhoc_mix += 2.0 * x_h2 * x_he * 1.0 * 1.0 * rhoc_cross(rhoc[0], rhoc[1])
    # H2-CH4
    inv_rhoc_mix += (
        2.0 * x_h2 * x_ch4 * (H2CH4_GAMMAV / H2CH4_BETAV) * rhoc_cross(rhoc[0], rhoc[2])
    )
    # H2-H2O
    inv_rhoc_mix += (
        2.0 * x_h2 * x_h2o * (H2H2O_GAMMAV / H2H2O_BETAV) * rhoc_cross(rhoc[0], rhoc[3])
    )
    # Others (Beta=Gamma=1)
    inv_rhoc_mix += 2.0 * x_he * x_ch4 * rhoc_cross(rhoc[1], rhoc[2])
    inv_rhoc_mix += 2.0 * x_he * x_h2o * rhoc_cross(rhoc[1], rhoc[3])
    inv_rhoc_mix += 2.0 * x_ch4 * x_h2o * rhoc_cross(rhoc[2], rhoc[3])

    if inv_rhoc_mix > 0:
        rhoc_mix_molL = 1.0 / inv_rhoc_mix
    else:
        rhoc_mix_molL = rhoc_h2_molL  # fallback

    return Tc_mix, rhoc_mix_molL


# ==============================================================================
# COMPRESSIBILITY FACTOR CALCULATION
# ==============================================================================


@njit(f64(f64, f64, f64, f64, f64, f64), cache=True)
def compute_Z_from_density(T, rho_molL, x_h2, x_he, x_ch4, x_h2o):
    """Compute compressibility factor Z at given temperature and molar density.

    Uses the GERG multi-fluid approach: all component Helmholtz energies are
    evaluated at mixture reduced variables (tau_mix, delta_mix). Includes
    binary departure functions for H2-CH4 and H2-H2O.

    Parameters
    ----------
    T : float
        Temperature in K.
    rho_molL : float
        Molar density of the mixture in mol/L.
    x_h2 : float
        Mole fraction of H2.
    x_he : float
        Mole fraction of He.
    x_ch4 : float
        Mole fraction of CH4.
    x_h2o : float
        Mole fraction of H2O.

    Returns
    -------
    float
        Compressibility factor Z (dimensionless).
    """
    Tc_mix, rhoc_mix_molL = mixture_reducing_parameters(x_h2, x_he, x_ch4, x_h2o)

    tau = Tc_mix / T
    delta = rho_molL / rhoc_mix_molL

    # All components evaluated at the SAME mixture reduced variables
    dalpha = 0.0

    if x_h2 > 0:
        h2_zeros = np.zeros(14)
        dalpha += x_h2 * d_alpha_d_delta_pure(
            tau,
            delta,
            H2_N_I,
            H2_T_I,
            H2_D_I,
            H2_P_I,
            H2_PHI_I,
            H2_BETA_I,
            H2_GAMMA_I,
            H2_D_GAUSS_I,
            H2_N_POW_NOEXP,
            H2_N_POW_WEXP,
            H2_N_GAUSS,
            H2_N_CRIT,
            h2_zeros,
            h2_zeros,
            h2_zeros,
            h2_zeros,
            h2_zeros,
            h2_zeros,
        )

    if x_he > 0:
        he_zeros = np.zeros(80)
        dalpha += x_he * d_alpha_d_delta_pure(
            tau,
            delta,
            HE_N_I,
            HE_T_I,
            HE_D_I,
            HE_P_I,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            HE_N_POW_NOEXP,
            HE_N_POW_WEXP,
            HE_N_GAUSS,
            HE_N_CRIT,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
        )

    if x_ch4 > 0:
        ch4_zeros = np.zeros(40)
        dalpha += x_ch4 * d_alpha_d_delta_pure(
            tau,
            delta,
            CH4_N_I,
            CH4_T_I,
            CH4_D_I,
            CH4_P_I,
            CH4_PHI_I,
            CH4_BETA_I,
            CH4_GAMMA_I,
            CH4_D_GAUSS_I,
            CH4_N_POW_NOEXP,
            CH4_N_POW_WEXP,
            CH4_N_GAUSS,
            CH4_N_CRIT,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
        )

    if x_h2o > 0:
        dalpha += x_h2o * d_alpha_d_delta_pure(
            tau,
            delta,
            H2O_N_I,
            H2O_T_I,
            H2O_D_I,
            H2O_P_I,
            H2O_PHI_I,
            H2O_BETA_I,
            H2O_GAMMA_I,
            H2O_D_GAUSS_I,
            H2O_N_POW_NOEXP,
            H2O_N_POW_WEXP,
            H2O_N_GAUSS,
            H2O_N_CRIT,
            H2O_RES_a,
            H2O_RES_b,
            H2O_RES_B,
            H2O_RES_C,
            H2O_RES_D,
            H2O_RES_A,
        )

    # Binary interaction terms (also at mixture tau, delta)
    if x_h2 > 0 and x_ch4 > 0:
        h2ch4_zeros = np.zeros(4)
        dalpha += (
            x_h2
            * x_ch4
            * d_alpha_d_delta_pure(
                tau,
                delta,
                H2CH4_N_I,
                H2CH4_T_I,
                H2CH4_D_I,
                H2CH4_P_I,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                H2CH4_N_POW_NOEXP,
                H2CH4_N_POW_WEXP,
                0,
                0,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
            )
        )

    if x_h2 > 0 and x_h2o > 0:
        h2h2o_zeros = np.zeros(4)
        dalpha += (
            x_h2
            * x_h2o
            * d_alpha_d_delta_pure(
                tau,
                delta,
                H2H2O_N_I,
                H2H2O_T_I,
                H2H2O_D_I,
                H2H2O_P_I,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                H2H2O_N_POW_NOEXP,
                H2H2O_N_POW_WEXP,
                0,
                0,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
            )
        )

    Z = 1.0 + delta * dalpha
    return Z


@njit(f64(f64, f64, f64, f64, f64, f64), cache=True)
def compute_Z(P, T, x_h2, x_he, x_ch4, x_h2o):
    """Compute compressibility factor Z given pressure, temperature, and composition.

    Uses fixed-point iteration on molar density: rho = P / (Z * R * T).
    Converges in 3-5 iterations for giant planet conditions (Z near 1).
    Trace gases (mole fraction < 0.01) are excluded because their binary
    interaction parameters are not calibrated at low concentrations.

    Parameters
    ----------
    P : float
        Total pressure in Pa.
    T : float
        Temperature in K.
    x_h2 : float
        Mole fraction of H2.
    x_he : float
        Mole fraction of He.
    x_ch4 : float
        Mole fraction of CH4.
    x_h2o : float
        Mole fraction of H2O.

    Returns
    -------
    float
        Compressibility factor Z (dimensionless).
    """
    R = 8.314462618  # J/(mol*K)

    # Only include gases with significant mole fraction in the Z calculation
    # Trace components (<1%) contribute negligibly to bulk compressibility
    # and their binary interaction parameters may not be valid at low concentrations
    MIN_X_FOR_EOS = 1e-6
    x_ch4_eff = x_ch4 if x_ch4 >= MIN_X_FOR_EOS else 0.0
    x_h2o_eff = x_h2o if x_h2o >= MIN_X_FOR_EOS else 0.0

    if P > 1e8:
        print("WARNING [compute_Z]: Pressure exceeds 1e8 Pa. Returning Z=1.")
        return 1.0

    # Fixed-point iteration: rho = P/(Z*R*T)
    rho_molL = P / (R * T) / 1000.0
    Z = 1.0
    for iteration in range(30):
        Z_new = compute_Z_from_density(T, rho_molL, x_h2, x_he, x_ch4_eff, x_h2o_eff)
        if Z_new <= 0.0 or Z_new > 100.0:
            print(
                "WARNING [compute_Z]: iteration diverging (Z=",
                Z_new,
                ") at P=",
                P / 1e5,
                "bar, T=",
                T,
                "K. Returning Z=1.",
            )
            return 1.0
        if abs(Z_new - Z) < CONVERGENCE_PRECISION:
            break
        Z = Z_new
        rho_molL = P / (Z * R * T) / 1000.0

    return Z


# ==============================================================================
# ISOBARIC HEAT CAPACITY (Cp) CALCULATION
# ==============================================================================


@njit(f64(f64, f64, f64, f64, f64, f64, f64, i64), cache=True)
def compute_Cp(P, T, Z, x_h2, x_he, x_ch4, x_h2o, h2_type):
    """Compute real-gas isobaric heat capacity Cp from the Helmholtz EOS.

    Uses the thermodynamic relation:
    Cp/R = -tau^2*(alpha0_tt + alphaR_tt)
           + (1 + delta*alphaR_d - delta*tau*alphaR_dt)^2
             / (1 + 2*delta*alphaR_d + delta^2*alphaR_dd)

    Parameters
    ----------
    P : float
        Total pressure in Pa.
    T : float
        Temperature in K.
    Z : float
        Compressibility factor, computed using compute_Z .
    x_h2 : float
        Mole fraction of H2.
    x_he : float
        Mole fraction of He.
    x_ch4 : float
        Mole fraction of CH4.
    x_h2o : float
        Mole fraction of H2O.
    h2_type : int
        H2 variant: 0 = normal, 1 = ortho, 2 = para.

    Returns
    -------
    float
        Molar isobaric heat capacity Cp in J/(mol*K).
    """
    R = 8.314462618

    MIN_X_FOR_EOS = 1e-6
    x_ch4_eff = x_ch4 if x_ch4 >= MIN_X_FOR_EOS else 0.0
    x_h2o_eff = x_h2o if x_h2o >= MIN_X_FOR_EOS else 0.0

    # EOS is not valid above ~1 kbar; return ideal Cp
    if P > 1e8:
        cp_ideal = 0.0
        if x_h2 > 0:
            cp_ideal += x_h2 * ideal_alpha_dtau_dtau_cp(H2_NI, H2_TC, H2_VI, H2_UI, T)
        if x_he > 0:
            cp_ideal += x_he * ideal_alpha_dtau_dtau_cp(HE_NI, HE_TC, HE_VI, HE_UI, T)
        if x_ch4_eff > 0:
            cp_ideal += x_ch4_eff * ideal_alpha_dtau_dtau_cp(
                CH4_NI, CH4_TC, CH4_VI, CH4_UI, T
            )
        if x_h2o_eff > 0:
            tau_h2o = H2O_TC / T
            alpha0_tt_h2o = ideal_alpha_dtau_dtau_coef(
                tau_h2o, H2O_IDEAL_N, H2O_IDEAL_GAMMA
            )
            cp_ideal += x_h2o_eff * (-(tau_h2o**2) * alpha0_tt_h2o + 1.0)
        return cp_ideal * R

    # Mixture reduced parameters
    Tc_mix, rhoc_mix_molL = mixture_reducing_parameters(
        x_h2, x_he, x_ch4_eff, x_h2o_eff
    )

    # Compute density from pre-computed Z
    rho_molL = P / (Z * R * T) / 1000.0
    tau = Tc_mix / T
    delta = rho_molL / rhoc_mix_molL

    # Select H2 parameters based on h2_type
    if h2_type == 1:
        h2_N = ORTHO_H2_N_I
        h2_T = ORTHO_H2_T_I
        h2_D = ORTHO_H2_D_I
        h2_P = ORTHO_H2_P_I
        h2_PHI = ORTHO_H2_PHI_I
        h2_BETA = ORTHO_H2_BETA_I
        h2_GAMMA = ORTHO_H2_GAMMA_I
        h2_DGAUSS = ORTHO_H2_D_GAUSS_I
        h2_tc = ORTHO_H2_TC
        h2_ni = ORTHO_H2_NI
        h2_vi = ORTHO_H2_VI
        h2_ui = ORTHO_H2_UI
    elif h2_type == 2:
        h2_N = PARA_H2_N_I
        h2_T = PARA_H2_T_I
        h2_D = PARA_H2_D_I
        h2_P = PARA_H2_P_I
        h2_PHI = PARA_H2_PHI_I
        h2_BETA = PARA_H2_BETA_I
        h2_GAMMA = PARA_H2_GAMMA_I
        h2_DGAUSS = PARA_H2_D_GAUSS_I
        h2_tc = PARA_H2_TC
        h2_ni = PARA_H2_NI
        h2_vi = PARA_H2_VI
        h2_ui = PARA_H2_UI
    else:
        h2_N = H2_N_I
        h2_T = H2_T_I
        h2_D = H2_D_I
        h2_P = H2_P_I
        h2_PHI = H2_PHI_I
        h2_BETA = H2_BETA_I
        h2_GAMMA = H2_GAMMA_I
        h2_DGAUSS = H2_D_GAUSS_I
        h2_tc = H2_TC
        h2_ni = H2_NI
        h2_vi = H2_VI
        h2_ui = H2_UI

    # --- Residual derivatives (mixture-tau, mixture-delta) ---
    dalpha_d = 0.0
    dalpha_tt = 0.0
    dalpha_dt = 0.0
    dalpha_dd = 0.0

    if x_h2 > 0:
        dalpha_d += x_h2 * d_alpha_d_delta_pure(
            tau,
            delta,
            h2_N,
            h2_T,
            h2_D,
            h2_P,
            h2_PHI,
            h2_BETA,
            h2_GAMMA,
            h2_DGAUSS,
            H2_N_POW_NOEXP,
            H2_N_POW_WEXP,
            H2_N_GAUSS,
            H2_N_CRIT,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
        )
        dalpha_tt += x_h2 * d_alpha_d_tau_d_tau_pure(
            tau,
            delta,
            h2_N,
            h2_T,
            h2_D,
            h2_P,
            h2_PHI,
            h2_BETA,
            h2_GAMMA,
            h2_DGAUSS,
            H2_N_POW_NOEXP,
            H2_N_POW_WEXP,
            H2_N_GAUSS,
            H2_N_CRIT,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
        )
        dalpha_dt += x_h2 * d_alpha_d_delta_d_tau_pure(
            tau,
            delta,
            h2_N,
            h2_T,
            h2_D,
            h2_P,
            h2_PHI,
            h2_BETA,
            h2_GAMMA,
            h2_DGAUSS,
            H2_N_POW_NOEXP,
            H2_N_POW_WEXP,
            H2_N_GAUSS,
            H2_N_CRIT,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
        )
        dalpha_dd += x_h2 * d_alpha_d_delta_d_delta_pure(
            tau,
            delta,
            h2_N,
            h2_T,
            h2_D,
            h2_P,
            h2_PHI,
            h2_BETA,
            h2_GAMMA,
            h2_DGAUSS,
            H2_N_POW_NOEXP,
            H2_N_POW_WEXP,
            H2_N_GAUSS,
            H2_N_CRIT,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
            h2_DGAUSS,
        )

    if x_he > 0:
        he_zeros = np.zeros(80)
        dalpha_d += x_he * d_alpha_d_delta_pure(
            tau,
            delta,
            HE_N_I,
            HE_T_I,
            HE_D_I,
            HE_P_I,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            HE_N_POW_NOEXP,
            HE_N_POW_WEXP,
            HE_N_GAUSS,
            HE_N_CRIT,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
        )
        dalpha_tt += x_he * d_alpha_d_tau_d_tau_pure(
            tau,
            delta,
            HE_N_I,
            HE_T_I,
            HE_D_I,
            HE_P_I,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            HE_N_POW_NOEXP,
            HE_N_POW_WEXP,
            HE_N_GAUSS,
            HE_N_CRIT,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
        )
        dalpha_dt += x_he * d_alpha_d_delta_d_tau_pure(
            tau,
            delta,
            HE_N_I,
            HE_T_I,
            HE_D_I,
            HE_P_I,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            HE_N_POW_NOEXP,
            HE_N_POW_WEXP,
            HE_N_GAUSS,
            HE_N_CRIT,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
        )
        dalpha_dd += x_he * d_alpha_d_delta_d_delta_pure(
            tau,
            delta,
            HE_N_I,
            HE_T_I,
            HE_D_I,
            HE_P_I,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            HE_N_POW_NOEXP,
            HE_N_POW_WEXP,
            HE_N_GAUSS,
            HE_N_CRIT,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
            he_zeros,
        )

    if x_ch4_eff > 0:
        ch4_zeros = np.zeros(40)
        dalpha_d += x_ch4_eff * d_alpha_d_delta_pure(
            tau,
            delta,
            CH4_N_I,
            CH4_T_I,
            CH4_D_I,
            CH4_P_I,
            CH4_PHI_I,
            CH4_BETA_I,
            CH4_GAMMA_I,
            CH4_D_GAUSS_I,
            CH4_N_POW_NOEXP,
            CH4_N_POW_WEXP,
            CH4_N_GAUSS,
            CH4_N_CRIT,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
        )
        dalpha_tt += x_ch4_eff * d_alpha_d_tau_d_tau_pure(
            tau,
            delta,
            CH4_N_I,
            CH4_T_I,
            CH4_D_I,
            CH4_P_I,
            CH4_PHI_I,
            CH4_BETA_I,
            CH4_GAMMA_I,
            CH4_D_GAUSS_I,
            CH4_N_POW_NOEXP,
            CH4_N_POW_WEXP,
            CH4_N_GAUSS,
            CH4_N_CRIT,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
        )
        dalpha_dt += x_ch4_eff * d_alpha_d_delta_d_tau_pure(
            tau,
            delta,
            CH4_N_I,
            CH4_T_I,
            CH4_D_I,
            CH4_P_I,
            CH4_PHI_I,
            CH4_BETA_I,
            CH4_GAMMA_I,
            CH4_D_GAUSS_I,
            CH4_N_POW_NOEXP,
            CH4_N_POW_WEXP,
            CH4_N_GAUSS,
            CH4_N_CRIT,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
        )
        dalpha_dd += x_ch4_eff * d_alpha_d_delta_d_delta_pure(
            tau,
            delta,
            CH4_N_I,
            CH4_T_I,
            CH4_D_I,
            CH4_P_I,
            CH4_PHI_I,
            CH4_BETA_I,
            CH4_GAMMA_I,
            CH4_D_GAUSS_I,
            CH4_N_POW_NOEXP,
            CH4_N_POW_WEXP,
            CH4_N_GAUSS,
            CH4_N_CRIT,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
            ch4_zeros,
        )

    if x_h2o_eff > 0:
        dalpha_d += x_h2o_eff * d_alpha_d_delta_pure(
            tau,
            delta,
            H2O_N_I,
            H2O_T_I,
            H2O_D_I,
            H2O_P_I,
            H2O_PHI_I,
            H2O_BETA_I,
            H2O_GAMMA_I,
            H2O_D_GAUSS_I,
            H2O_N_POW_NOEXP,
            H2O_N_POW_WEXP,
            H2O_N_GAUSS,
            H2O_N_CRIT,
            H2O_RES_a,
            H2O_RES_b,
            H2O_RES_B,
            H2O_RES_C,
            H2O_RES_D,
            H2O_RES_A,
        )
        dalpha_tt += x_h2o_eff * d_alpha_d_tau_d_tau_pure(
            tau,
            delta,
            H2O_N_I,
            H2O_T_I,
            H2O_D_I,
            H2O_P_I,
            H2O_PHI_I,
            H2O_BETA_I,
            H2O_GAMMA_I,
            H2O_D_GAUSS_I,
            H2O_N_POW_NOEXP,
            H2O_N_POW_WEXP,
            H2O_N_GAUSS,
            H2O_N_CRIT,
            H2O_RES_a,
            H2O_RES_b,
            H2O_RES_B,
            H2O_RES_C,
            H2O_RES_D,
            H2O_RES_A,
        )
        dalpha_dt += x_h2o_eff * d_alpha_d_delta_d_tau_pure(
            tau,
            delta,
            H2O_N_I,
            H2O_T_I,
            H2O_D_I,
            H2O_P_I,
            H2O_PHI_I,
            H2O_BETA_I,
            H2O_GAMMA_I,
            H2O_D_GAUSS_I,
            H2O_N_POW_NOEXP,
            H2O_N_POW_WEXP,
            H2O_N_GAUSS,
            H2O_N_CRIT,
            H2O_RES_a,
            H2O_RES_b,
            H2O_RES_B,
            H2O_RES_C,
            H2O_RES_D,
            H2O_RES_A,
        )
        dalpha_dd += x_h2o_eff * d_alpha_d_delta_d_delta_pure(
            tau,
            delta,
            H2O_N_I,
            H2O_T_I,
            H2O_D_I,
            H2O_P_I,
            H2O_PHI_I,
            H2O_BETA_I,
            H2O_GAMMA_I,
            H2O_D_GAUSS_I,
            H2O_N_POW_NOEXP,
            H2O_N_POW_WEXP,
            H2O_N_GAUSS,
            H2O_N_CRIT,
            H2O_RES_a,
            H2O_RES_b,
            H2O_RES_B,
            H2O_RES_C,
            H2O_RES_D,
            H2O_RES_A,
        )

    # Binary departure terms
    if x_h2 > 0 and x_ch4_eff > 0:
        h2ch4_zeros = np.zeros(4)
        dalpha_d += (
            x_h2
            * x_ch4_eff
            * d_alpha_d_delta_pure(
                tau,
                delta,
                H2CH4_N_I,
                H2CH4_T_I,
                H2CH4_D_I,
                H2CH4_P_I,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                H2CH4_N_POW_NOEXP,
                H2CH4_N_POW_WEXP,
                0,
                0,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
            )
        )
        dalpha_tt += (
            x_h2
            * x_ch4_eff
            * d_alpha_d_tau_d_tau_pure(
                tau,
                delta,
                H2CH4_N_I,
                H2CH4_T_I,
                H2CH4_D_I,
                H2CH4_P_I,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                H2CH4_N_POW_NOEXP,
                H2CH4_N_POW_WEXP,
                0,
                0,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
            )
        )
        dalpha_dt += (
            x_h2
            * x_ch4_eff
            * d_alpha_d_delta_d_tau_pure(
                tau,
                delta,
                H2CH4_N_I,
                H2CH4_T_I,
                H2CH4_D_I,
                H2CH4_P_I,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                H2CH4_N_POW_NOEXP,
                H2CH4_N_POW_WEXP,
                0,
                0,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
            )
        )
        dalpha_dd += (
            x_h2
            * x_ch4_eff
            * d_alpha_d_delta_d_delta_pure(
                tau,
                delta,
                H2CH4_N_I,
                H2CH4_T_I,
                H2CH4_D_I,
                H2CH4_P_I,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                H2CH4_N_POW_NOEXP,
                H2CH4_N_POW_WEXP,
                0,
                0,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
                h2ch4_zeros,
            )
        )

    if x_h2 > 0 and x_h2o_eff > 0:
        h2h2o_zeros = np.zeros(4)
        dalpha_d += (
            x_h2
            * x_h2o_eff
            * d_alpha_d_delta_pure(
                tau,
                delta,
                H2H2O_N_I,
                H2H2O_T_I,
                H2H2O_D_I,
                H2H2O_P_I,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                H2H2O_N_POW_NOEXP,
                H2H2O_N_POW_WEXP,
                0,
                0,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
            )
        )
        dalpha_tt += (
            x_h2
            * x_h2o_eff
            * d_alpha_d_tau_d_tau_pure(
                tau,
                delta,
                H2H2O_N_I,
                H2H2O_T_I,
                H2H2O_D_I,
                H2H2O_P_I,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                H2H2O_N_POW_NOEXP,
                H2H2O_N_POW_WEXP,
                0,
                0,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
            )
        )
        dalpha_dt += (
            x_h2
            * x_h2o_eff
            * d_alpha_d_delta_d_tau_pure(
                tau,
                delta,
                H2H2O_N_I,
                H2H2O_T_I,
                H2H2O_D_I,
                H2H2O_P_I,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                H2H2O_N_POW_NOEXP,
                H2H2O_N_POW_WEXP,
                0,
                0,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
            )
        )
        dalpha_dd += (
            x_h2
            * x_h2o_eff
            * d_alpha_d_delta_d_delta_pure(
                tau,
                delta,
                H2H2O_N_I,
                H2H2O_T_I,
                H2H2O_D_I,
                H2H2O_P_I,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                H2H2O_N_POW_NOEXP,
                H2H2O_N_POW_WEXP,
                0,
                0,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
                h2h2o_zeros,
            )
        )

    # --- Ideal gas contribution: -tau_i^2 * d²(alpha0_i)/d(tau_i²) ---
    # Each component uses its own tau_i = Tc_i / T for the ideal part.
    # ideal_alpha_dtau_dtau_cp returns Cp0/R directly (= ni + planck_einstein sum).
    # We need -tau_i^2 * alpha0_tt_i = Cp0/R - 1 (the Cv_ideal/R contribution).
    neg_tau2_alpha0_tt = 0.0

    if x_h2 > 0:
        neg_tau2_alpha0_tt += x_h2 * (
            ideal_alpha_dtau_dtau_cp(h2_ni, h2_tc, h2_vi, h2_ui, T) - 1.0
        )

    if x_he > 0:
        neg_tau2_alpha0_tt += x_he * (
            ideal_alpha_dtau_dtau_cp(HE_NI, HE_TC, HE_VI, HE_UI, T) - 1.0
        )

    if x_ch4_eff > 0:
        neg_tau2_alpha0_tt += x_ch4_eff * (
            ideal_alpha_dtau_dtau_cp(CH4_NI, CH4_TC, CH4_VI, CH4_UI, T) - 1.0
        )

    if x_h2o_eff > 0:
        tau_h2o = H2O_TC / T
        alpha0_tt_h2o = ideal_alpha_dtau_dtau_coef(
            tau_h2o, H2O_IDEAL_N, H2O_IDEAL_GAMMA
        )
        neg_tau2_alpha0_tt += x_h2o_eff * (-(tau_h2o**2) * alpha0_tt_h2o)

    # --- Assemble Cp ---
    # Cp/R = -tau^2*(alpha0_tt + alphaR_tt) + (1+d*aR_d - d*t*aR_dt)^2 / (1+2*d*aR_d+d^2*aR_dd)
    # neg_tau2_alpha0_tt already holds the -tau_i^2*alpha0_tt_i part per component
    # Cv_residual/R = -tau_mix^2 * alphaR_tt (using mixture tau)
    cv_residual_over_R = -(tau**2) * dalpha_tt

    numerator = (1.0 + delta * dalpha_d - delta * tau * dalpha_dt) ** 2
    denominator = 1.0 + 2.0 * delta * dalpha_d + delta**2 * dalpha_dd

    if denominator <= 0.0:
        print(
            "WARNING [compute_Cp]: mechanical instability (denominator=",
            denominator,
            ") at P=",
            P / 1e5,
            "bar, T=",
            T,
            "K. Returning ideal Cp.",
        )
        return (neg_tau2_alpha0_tt + 1.0) * R

    cp_over_R = neg_tau2_alpha0_tt + cv_residual_over_R + numerator / denominator

    if cp_over_R < 1.5 or cp_over_R > 15.0:
        print(
            "WARNING [compute_Cp]: unphysical Cp/R=",
            cp_over_R,
            " at P=",
            P / 1e5,
            "bar, T=",
            T,
            "K (delta=",
            delta,
            ", tau=",
            tau,
            "). Returning ideal Cp.",
        )
        return (neg_tau2_alpha0_tt + 1.0) * R

    return cp_over_R * R
