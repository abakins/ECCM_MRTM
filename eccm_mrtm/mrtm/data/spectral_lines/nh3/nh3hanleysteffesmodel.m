function alphanh3=nh3hanleysteffesmodel(f,T,P,H2mr,Hemr,NH3mr) % Hanley-Steffes model from 2008 PhD thesis

%input f as a vector in GHz, T in Kelvin, P in Bars, H2mr, Hemr and NH3mr in mole
%fraction, so for a 1% mixture of NH3, NH3mr = 0.01, etc.  f is the only
%variable that can be a vector.  Opacity (alphanh3) is returned as a vector matching f
%in dB/km.

load nh3lincat190Latest;	
%Loads the Poynter Pickett (JPL) nh3 line catalog.  The workspace contains the arrays for Freq (fo) in GHz, intensity (Io) in inverse cm,
%lower state energy (Eo) in inverse cm and self broadening due to NH3 (gammaNH3o) in GHz/bar.
%Constants
GHztoinv_cm=1/29.9792458;	            %for converting GHz to inverse cm
OpticaldepthstodB=434294.5;				%converts from cm^-1 to dB/km
torrperatm=760;
bartoatm=0.987;
GHztoMHz=1000;
hc=19.858252418E-24;			%planks (J.s) light (cm/s)
k=1.38*10^-23;					%boltzmann's in J/K or N.m/K
No=6.02297e23;					%Avogadros Number [mole^-1]
R=8.31432e7;					%Rydberg's [erg/mole-K]
To=300;			                %Ref temp for P/P Catalogue
dynesperbar=1e6;				%dyne=bar/1e6;
coef=dynesperbar*No/R;          

PH2=P*H2mr; %partial pressures
PHe=P*Hemr;
PNH3=P*NH3mr;

%calculate vector linewidth
xi1=0.7756; 
xi2=2/3;
xi3=1;
xi12=0.7964; 
xi22=2/3; 
xi32=1.554;
Tdiv=To/T;
gnu1=1.640; 
gnu2=0.75; 
gnu3=0.852; 
gH2=gnu1*PH2;
gHe=gnu2*PHe;
gNH3=gnu3*PNH3*gammaNH3o;
gamma=(gH2)*((Tdiv)^(xi1))+(gHe)*((Tdiv)^(xi2))+gNH3*(295/T)^(xi3);

delt=-0.0498*gamma; 
znu1=1.262;
znu2=0.3; 
znu3=0.5296; 
zH2=znu1*PH2;
zHe=znu2*PHe;
zNH3=znu3*PNH3*gammaNH3o;
zeta=(zH2)*((Tdiv)^(xi12))+(zHe)*((Tdiv)^(xi22))+zNH3*(295/T)^(xi32);

zetasize=size(fo,1);
pst=delt;		% answer in GHz
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

eta=3/2;			% for symmetric top molecule
expo=-(1/T-1/To)*Eo*hc/k;
ST=Io.*exp(expo);	% S(T) =S(To)converted for temperature
Con=0.9301;
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

alpha_opdep=sum(br_alpha_matrix,1);

%sums up the element in the matrix to calculate the alpha in optical depths or inverse cm
%alpha_opdep=sum(br_alpha_matrix,1);


alphanh3=alpha_opdep*434294.5;
%answer in inverse cm converted to dB/km


