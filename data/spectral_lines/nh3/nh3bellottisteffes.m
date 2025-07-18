%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS FUNCTION COMPUTES AMMONIA ABSORPTION UNDER JOVIAN ATMOSPHERIC
% CONDITIONS BETWEEN 1-150 GHz
%
% NAME:
%      NH3H2HeModel
%
% EXPLANATION:
%   This function can be used to compute the opacity of pure ammonia and
%   that of ammonia in a hydrogen/helium atmosphere between 1-150 GHz under
%   jovian conditions
%
% CALLING SEQUENCE:
%       alphanh3=NH3H2HeModel(f,T,P,H2mr,Hemr,NH3mr)
%
% INPUTS:
%       f        - Array of frequencies (GHz)
%       T        - Temperature (K)
%       P        - Pressure (bars)
%       H2mr     - Hydrogen mixing ratio (as mole fraction)
%       Hemr     - Helium mixing ratio (as mole fraction)
%       NH3mr    - Ammonia mixing ratio (as mole fraction)
%
% OUTPUTS:
%      alphanh3  - Array of ammonia opacity (dB/km) computed at the input frequencies
%
% METHOD:
%   A modified Ben Reuven (Ben Reuven, 1966) lineshape is used for computing
%   ammonia opacity due to inversion lines, and a Gross lineshape (Gross, 1955)
%   is used for computing ammonia opacity due to the rotational lines and
%   the v2 roto-vibrational lines.
%
% History:
%       written by Kiruthika Devaraj at Stanford,  April, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Function Unified_NH3_Model
function alphanh3=NH3H2HeModel_bell_quick(f,T,P,H2mr,Hemr,NH3mr,CH4mr,H2Omr,fo, Io, Eo, gammaNH3o, H2HeBroad,fo_rot, Io_rot, Eo_rot, gNH3_rot, gH2_rot, gHe_rot,fo_v2, Io_v2, Eo_v2)

% The data files containing the frequency, line intensity and lower state 
% energy for the ammonia transitions as given in the latest JPL spectral 
% line catalog provided by Shanshan Yu and Brian Droiun  (personal 
% communication, 2010), and the self and foreign gas broadening 
% parameters for the various transitions as given by Devaraj et al. Icarus,
% 2010 are loaded in the following steps. 

% % Inversion lines: fo is frequency in GHz, Io is line intensity in
% % cm^-1/(molecule./cm^2), Eo is lower state energy in cm^-1, gammaNH3o and
% % H2HeBroad are self and foreign gas broadening parameters.
% [fo, Io, Eo, gammaNH3o, H2HeBroad] = textread('ammonia_inversion.dat','%f %f %f %f %f','headerlines',1);
% 
% % Rotational lines: fo_rot is frequency in GHz, Io_rot is line intensity in
% % cm^-1/(molecule./cm^2), Eo_rot is lower state energy in cm^-1, gNH3_rot,
% % gH2_rot, gHe_rot are broadening parameters for rotational lines.
% [fo_rot, Io_rot, Eo_rot, gNH3_rot, gH2_rot gHe_rot] = textread('ammonia_rotational.dat','%f %f %f %f %f %f','headerlines',1);
% 
% % v2 roto-vibrational lines: fo_v2 is frequency in GHz, Io_v2 is line intensity in
% % cm^-1/(molecule./cm^2), Eo_v2 is lower state energy in cm^-1,
% [fo_v2, Io_v2, Eo_v2] = textread('ammonia_rotovibrational.dat','%f %f %f','headerlines',1);

GHztoinv_cm=1/29.9792458;           % for converting GHz to inverse cm
OpticaldepthstodB=434294.5;			% convert from cm^-1 to dB/km
torrperatm=760;                     % convert from atm to torr
bartoatm=0.987;                     % convert from bat to atm
GHztoMHz=1000;                      % convert from GHz to MHz
hc=19.858252418E-24;                %planks (J.s) light (cm/s)
k=1.38*10^-23;                      %boltzmann's in J/K or N.m/K
No=6.02297e23;                      %Avogadros Number [mole^-1]
R=8.31432e7;                        %Rydberg's [erg/mole-K]
To=300;                             %Ref temp for P/P Catalogue
dynesperbar=1e6;                    %dyne=bar/1e6;
coef=dynesperbar*No/R;              %See Appendix D: Using the Poyter-Pickett Catalogs

% Compute the partial pressure of H2, He, and NH2
PH2=P*H2mr;
PHe=P*Hemr;
PNH3=P*NH3mr;
PCH4=P*CH4mr;
PH2O=P*H2Omr;

LowFreq = f<30;

% Compute the temperature factor
Tdiv=To/T;

%  Coefficient for symmetric top molecule
eta=3/2;
%% Computing the opacity due to the inversion lines

% Pressure Dependent Switch for the parameters of the inversion transitions
% for low centimeter-wave frequency
    gnu_H2=1.6937;          gnu_He=0.6997;          gnu_NH3=0.7523;
    GAMMA_H2= 0.8085;       GAMMA_He=1;             GAMMA_NH3=1;
    zeta_H2=1.3263;         zeta_He=0.1607;         zeta_NH3=0.6162;
    Z_H2= 0.8199;           Z_He=-0.7269;           Z_NH3=1.3832;
    d=-0.0139;
    Con=0.9619;

gnu_CH4 = 3.32366689;
GAMMA_CH4 = 0.5;
zeta_CH4 = 0.244756;
Z_CH4 = 0.08028144;

gnu_H2O = 5.31188294;
GAMMA_H2O = 0.622369;
zeta_H2O = 5.23336187;
Z_H2O = 2.12484818;     
% Individual broadening parameters
gH2=gnu_H2*PH2; gHe=gnu_He*PHe; gNH3=gnu_NH3*PNH3*gammaNH3o; gCH4=gnu_CH4*PCH4;gH2O=gnu_H2O*PH2O;
% Broadening parameter
gamma=((gH2)*((Tdiv)^(GAMMA_H2))+(gHe)*((Tdiv)^(GAMMA_He))+gNH3*(295/T)^(GAMMA_NH3)) +(gCH4)*((Tdiv)^(GAMMA_CH4))+(gH2O)*((Tdiv)^(GAMMA_H2O));

% Shift parameter
delt=d*gamma;

% Individual coupling parameters
zH2=zeta_H2*PH2;  zHe=zeta_He*PHe; zNH3=zeta_NH3*PNH3*gammaNH3o; zCH4 = zeta_CH4*PCH4; zH2O = zeta_H2O*PH2O;
% Coupling parameter
zeta=(zH2)*((Tdiv)^(Z_H2))+(zHe)*((Tdiv)^(Z_He))+zNH3*(295/T)^(Z_NH3)+(zCH4)*((Tdiv)^(Z_CH4))+(zH2O)*((Tdiv)^(Z_H2O));

zetasize=size(fo,1);
pst=delt;       							% answer in GHz
%Coupling element, pressure shift and dnu or gamma are in GHz, need to convert brlineshape to inverse cm which is done below

n=size(f,2);  %returns the number of columns in f
m=size(fo,1); %returns the number of rows in fo
% f1 f2 f3 f4 ....fn  n times where n is the number of frequency steps
% f1 f2 f3 f4 ....fn				in the observation range
% ...
% f1 f2 f3 f4 ....fn
% m times where m is the number of spectral lines

nones=ones(1,n);
mones=ones(m,1);
f_matrix=mones*f;
fo_matrix=fo*nones;

% The 10^6 allows use of P(bar) for P(dynes/cm^2)
expo=-(1/T-1/To)*Eo*hc/k;
ST=Io.*exp(expo);	% S(T) =S(To)converted for temperature
alpha_noshape=Con*coef*(PNH3/To)*((To/T)^(eta+2)).*ST;%0.9387
%Alpha Max Found

%Ben Reuven lineshape calculated by the brlineshape function gives the answer in GHz
%Here we change from GHz to inverse cm.
dnu_matrix=gamma*nones;
ce_matrix=zeta*nones;
pst_matrix=pst*nones;
Aa=(2/pi)*((f_matrix./fo_matrix).^2);
Bb=(dnu_matrix-ce_matrix).*(f_matrix.^2);
Cc=dnu_matrix+ce_matrix;
Dd=((fo_matrix+pst_matrix).^2) + (dnu_matrix.^2)-(ce_matrix.^2);
Ee=f_matrix.^2;
Jj=(fo_matrix+pst_matrix).^2;
Gg=dnu_matrix.^2;
Hh=ce_matrix.^2;
Ii=4*(f_matrix.^2).*(dnu_matrix.^2);
Ff=Aa.*(Bb+Cc.*Dd)./(((Ee-Jj-Gg+Hh).^2)+Ii);

Fbr=(1/GHztoinv_cm).*Ff;
alpha_noshape_matrix=alpha_noshape*nones;
br_alpha_matrix_low_freq=alpha_noshape_matrix.*Fbr;


% Pressure Dependent Switch for the parameters of the inversion transitions
% for millimeter wave frequency
    gnu_H2=1.6361;          gnu_He=0.4555;          gnu_NH3=0.7298;
    GAMMA_H2= 0.8;       GAMMA_He=5;             GAMMA_NH3=1;
    zeta_H2=1.1313;         zeta_He=0.1;         zeta_NH3=0.5152;
    Z_H2= 0.6234;           Z_He=-0.5;           Z_NH3=2/3;
    d=0.2;
    Con=1.3746;

% Individual broadening parameters
gH2=gnu_H2*PH2; gHe=gnu_He*PHe; gNH3=gnu_NH3*PNH3*gammaNH3o;
% Broadening parameter
gamma=((gH2)*((Tdiv)^(GAMMA_H2))+(gHe)*((Tdiv)^(GAMMA_He))+gNH3*(295/T)^(GAMMA_NH3));

% Shift parameter
delt=d*gamma;

% Individual coupling parameters
zH2=zeta_H2*PH2;  zHe=zeta_He*PHe; zNH3=zeta_NH3*PNH3*gammaNH3o;
% Coupling parameter
zeta=(zH2)*((Tdiv)^(Z_H2))+(zHe)*((Tdiv)^(Z_He))+zNH3*(295/T)^(Z_NH3);

zetasize=size(fo,1);
pst=delt;       							% answer in GHz
%Coupling element, pressure shift and dnu or gamma are in GHz, need to convert brlineshape to inverse cm which is done below

n=size(f,2);  %returns the number of columns in f
m=size(fo,1); %returns the number of rows in fo
% f1 f2 f3 f4 ....fn  n times where n is the number of frequency steps
% f1 f2 f3 f4 ....fn				in the observation range
% ...
% f1 f2 f3 f4 ....fn
% m times where m is the number of spectral lines

nones=ones(1,n);
mones=ones(m,1);
f_matrix=mones*f;
fo_matrix=fo*nones;

% The 10^6 allows use of P(bar) for P(dynes/cm^2)
expo=-(1/T-1/To)*Eo*hc/k;
ST=Io.*exp(expo);	% S(T) =S(To)converted for temperature
alpha_noshape=Con*coef*(PNH3/To)*((To/T)^(eta+2)).*ST;%0.9387
%Alpha Max Found

%Ben Reuven lineshape calculated by the brlineshape function gives the answer in GHz
%Here we change from GHz to inverse cm.
dnu_matrix=gamma*nones;
ce_matrix=zeta*nones;
pst_matrix=pst*nones;
Aa=(2/pi)*((f_matrix./fo_matrix).^2);
Bb=(dnu_matrix-ce_matrix).*(f_matrix.^2);
Cc=dnu_matrix+ce_matrix;
Dd=((fo_matrix+pst_matrix).^2) + (dnu_matrix.^2)-(ce_matrix.^2);
Ee=f_matrix.^2;
Jj=(fo_matrix+pst_matrix).^2;
Gg=dnu_matrix.^2;
Hh=ce_matrix.^2;
Ii=4*(f_matrix.^2).*(dnu_matrix.^2);
Ff=Aa.*(Bb+Cc.*Dd)./(((Ee-Jj-Gg+Hh).^2)+Ii);

Fbr=(1/GHztoinv_cm).*Ff;
alpha_noshape_matrix=alpha_noshape*nones;
br_alpha_matrix=alpha_noshape_matrix.*Fbr;

%% Computing the opacity due to rotational lines
% Computing the absorption contributed by the rotational lines
ST_rot=Io_rot.*(exp((1/To-1/T)*Eo_rot*hc/k));
% Factor GAMMA:
eta_H2=0.8730; eta_He=2/3; eta_NH3=1;
% Factor nu:
gnu_H2=0.2984; gnu_He=0.75; gnu_NH3=3.1789;
% Factor Con
Con=2.42688;

% Individual broadening
gH2=gnu_H2.*PH2.*gH2_rot; gHe=gnu_He.*PHe.*gHe_rot;
gNH3=gnu_NH3.*PNH3.*gNH3_rot;
% Total broadening
gamma_rot=((gH2)*((Tdiv)^(eta_H2))+(gHe)*((Tdiv)^(eta_He))+gNH3*(Tdiv)^(eta_NH3));
n=size(f,2);  %returns the number of columns in f
m=size(fo_rot,1);
nones=ones(1,n);
mones=ones(m,1);
f_matrix=mones*f;
fo_matrix=fo_rot*nones;
dnu_matrix=gamma_rot*nones;

% Gross Lineshape
Aa=4/pi*(f_matrix.^2).*dnu_matrix;
Bb=(fo_matrix.^2-f_matrix.^2).^2;
Cc=4*f_matrix.^2.*dnu_matrix.^2;
F_rot=Aa./(Bb+Cc);
Fbr_rot=(1/GHztoinv_cm).*F_rot;
alpha_rot=Con*coef*(PNH3/To)*((To/T)^(eta+2))*ST_rot*nones.*Fbr_rot;

%% Computing the opacity due to v2 roto-vibrational lines

% Computing the absorption contributed by the v2 rotovibrational lines
ST_v2=Io_v2.*(exp((1/To-1/T)*Eo_v2*hc/k));

% Broadening parameters for the v2 vibrational lines
gH2_v2=linspace(1.4,1.4,length(fo_v2))';
gHe_v2=linspace(0.68,0.68,length(fo_v2))';
gNH3_v2=linspace(9.5,9.5,length(fo_v2))';

% Factor GAMMA:
eta_H2=0.73; eta_He=0.5716; eta_NH3=1;
% Factor Con
Con=1.1206;

% Individual broadening parameters
gH2=PH2*gH2_v2; gHe=PHe*gHe_v2;gNH3=PNH3*gNH3_v2;
%Total broadening
gamma_v2=((gH2)*((Tdiv)^(eta_H2))+(gHe)*((Tdiv)^(eta_He))+gNH3*(Tdiv)^(eta_NH3));

n=size(f,2);  %returns the number of columns in f
m=size(fo_v2,1);
nones=ones(1,n);
mones=ones(m,1);
f_matrix=mones*f;
fo_matrix=fo_v2*nones;
dnu_matrix=gamma_v2*nones;

% Gross Lineshape
Aa=4/pi*(f_matrix.^2).*dnu_matrix;
Bb=(fo_matrix.^2-f_matrix.^2).^2;
Cc=4*f_matrix.^2.*dnu_matrix.^2;
F_v2=Aa./(Bb+Cc);
Fbr_v2=(1/GHztoinv_cm).*F_v2;
alpha_v2=Con*coef*(PNH3/To)*((To/T)^(eta+2))*ST_v2*nones.*Fbr_v2;



%% Computing the total opacity
%alpha_opdep=sum(br_alpha_matrix,1);
alpha_opdep_low=sum(br_alpha_matrix_low_freq,1)+sum(alpha_rot,1)+sum(alpha_v2,1);
alpha_opdep_high=sum(br_alpha_matrix,1)+sum(alpha_rot,1)+sum(alpha_v2,1);

HighFreq = f>=30;
alpha_opdep = alpha_opdep_low.*LowFreq + alpha_opdep_high.*HighFreq;


alphanh3=alpha_opdep*434294.5 ;%+ alpha_cont*4.342945;