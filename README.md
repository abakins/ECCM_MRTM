# ECCM - MRTM 

![Saturn ECCM](demos/Akins_Hofstadter_2026/Fig1_Fig3_ECCMDemo/saturn_eccm.pdf)

## ECCM 
ECCM is a simple equilibrium cloud condensation model relevant to computing the atmospheric composition and temperature profiles of giant planet tropospheres. 
It includes routines for computing profiles for water, ammonia, hydrogen sulfide, methane, and phosphine. 
Temperatures profiles are computed using either the dry adiabatic lapse rate or the wet adiabat when condensation occurs. 
Tropopause and stratosphere a priori profiles are included from radio occultation measurements. 

### Procedures 
- eccm - High-level interface 
- core - Core ECCM routines
- thermo - Vapor pressure and heat information for different gases 
- eos - Equation of state for H2/He/H2O/CH4 mixtures

## MRTM
MRTM is a simple non-scattering microwave radiative transfer model for computing microwave brightness temperatures from ECCM atmospheres. 

### Procedures 
- mrtm - Core MRTM routines
- molecule - Microwave opacity models for gases 

  


