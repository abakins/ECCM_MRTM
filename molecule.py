import numpy as np 
from numpy import exp, log10, pi
import scipy.constants as spc
import scipy.io as sio
import scipy.interpolate as spi
import os 

class molecule():
    """ Base molecule class for neutral gases which is used to calculate opacities.
        General methods are established for calculating line-by-line opacities associated with atmospheric constituents
        Inheriting classes relying on these functions should specify the following attributes
        lineshape_type: Either Van Vleck-Weisskopf, Gross, or Ben-Reuven
        lines and line_labels: Files containing and indexing spectral line parameters (see the import jpl file function) 
        parameters and parameter_labels: Files containing and indexing line-by-line broadening parameters for all relevant gases
        See one of the inheriting molecules defined below and the associated files in the spectral_lines folder to get a sense for how 
        to create your own inheriting molecule if necessary
    """ 

    To = 300
    hc = spc.h * spc.c * 100
    OpticaldepthstodBperkm = 10 * log10(exp(1)) * 1e5
    L = 2.6867811e19  # Lochschmidt's number for amagat conversion

    # Define some paths 
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    line_path = os.path.join(data_path, 'spectral_lines')
    jpl_path = os.path.join(data_path, 'spectral_lines', 'jpl_cat_files')
    lineshape_path = os.path.join(data_path, 'spectral_lines', 'lineshape_parameters')
    cia_path = os.path.join(data_path, 'absorption_tables')

    def __init__(self, **kwargs): 

        for key, value in kwargs.items(): 
            setattr(self, key, value) 

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'): 
        """ Calculates line-by-line microwave or millimeter wavelength opacity of the constituent of interest for a set of atmospheric conditions. 

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')

            Returns the absorption of the gas (Size MxN)

            Inheriting classes that either don't rely on a spectral line opacity calculation or require modifications should overwrite this function"""

        gas_pressure = gases * pressure * spc.bar / spc.torr
        gamma, zeta, delta = self.load_line_parameters(temperature, gas_pressure, gases_index)
        alpha_max = self.absorption_coeff(10**self.lines['LGINT'].ravel(), temperature, self.lines['ELO'].ravel(), 
                                          gases[gases_index.index(self.__class__.__name__), :] * pressure * spc.bar / spc.torr, gamma)
        if self.lineshape_type == "Van Vleck-Weisskopf":
            lineshape = self.vanvleckweisskopf(frequency, self.lines['FREQ'].ravel(), gamma)
        elif self.lineshape_type == "Gross":
            lineshape = self.gross(frequency, self.lines['FREQ'].ravel(), gamma)
        elif self.lineshape_type == "Ben-Reuven":
            lineshape = self.benreuven(frequency, self.lines['FREQ'].ravel(), gamma, zeta, delta)
        else:
            raise ValueError("Unspecified lineshape type. Valid options are 'Van Vleck-Weisskopf', 'Gross', or 'Ben-Reuven'") 
        absorption = np.sum(alpha_max * spc.pi * gamma * lineshape, axis=-1) 

        if units == 'dBperkm': 
            return absorption * self.OpticaldepthstodBperkm 
        elif units == 'invcm': 
            return absorption
        else:
            raise ValueError('Specify valid units (invcm or dBperkm')

    def load_line_parameters(self, temperature, gas_pressure, gases_index):
        """ Calculates linewidths from parameter tables and atmospheric conditions. In the literature, the linewidth gamma may also be written as Delta nu. 

            :param temperature: Temperature in Kelvins of the gas mixture (Size M)
            :param gas_pressure: Matrix of gas pressures in torr (Size NxM)
            :param gases_index: row indexing for the pressure array Size(N)

            Returns line parameters (Size OxN) where O is the number of spectral lines for the molecule

        """ 
        gamma = 0
        zeta = 0
        delta = 0
        index = [x for x in gases_index if x in self.parameters.keys()]
        for i in index: 
            gamma += self.linewidth_func(self.parameters[i][:, self.parameter_labels.index('GAMMA')], gas_pressure[gases_index.index(i)], self.To / temperature, self.parameters[i][:, self.parameter_labels.index('GAMMA_TEMP')])

        if self.lineshape_type == 'Ben-Reuven': 
            
            for i in index: 
                zeta += self.linewidth_func(self.parameters[i][:, self.parameter_labels.index('ZETA')], gas_pressure[gases_index.index(i)], self.To / temperature, self.parameters[i][:, self.parameter_labels.index('ZETA_TEMP')])
            delta = self.parameters[i][:, self.parameter_labels.index('DELTA')] * gamma
        return gamma, zeta, delta 

    def linewidth_func(self, width, pressure, theta, temp_coeff): 
        """ Linewidth calculation

            :param width: Line-by-line width parameters
            :param pressure: Pressure in torr
            :param theta: Temperature ratio e.g. 300/temperature
            :param temp_coeff: Line-by-line temperature coefficient

            For broadcasting to work, all inputs must be flattened 
        """
        w = width
        p = pressure[..., np.newaxis]
        t = theta[..., np.newaxis]
        tc = temp_coeff

        wide = w * p * t**tc
        return wide

    def absorption_coeff(self, intensity, temperature, lower_energy, torr, gamma):
        """ Computes line center absorption for each specified temperature and pressure.
        The initial intensity provided by the JPL catalog is specified at 300 K, and pressure is specified in torr
        For a linear molecule, the inverse dependence of intensity with temperature is 1, 1.5 for non-linear molecules 
        Output units are in inverse centimeters, and this is why the Planck constant and speed of light term are included despite 
        being absent from the expression in the line catalog documentation.

        :param intensity: Initial line intensity nm^2MHz (length M) (note: JPL catalog provides this as log(nm^2MHz))
        :param temperature: Gas temperature in Kelvin (length N)
        :param lower_energy: Energy of the lower state in the line transition in cm^-1 (length M)
        :param torr: Gas pressure in torr (length N)
        :param gamma: Half width at half maximum of the line in MHz (Size MxN)

        Returns line center absorption in cm^-1 (Size MxN)"""

        if self.linear is True:
            n_j = 1
        else: 
            n_j = 1.5

        intens = intensity
        t = temperature[..., np.newaxis]
        elo = lower_energy
        tor = torr[..., np.newaxis]

        lca = 102.458 * self.To / t * tor / gamma * intens * (self.To / t)**(n_j + 1) * exp(-(elo * self.hc / spc.k * (1 / t - 1 / self.To)))
        return lca

    @staticmethod
    def vanvleckweisskopf(frequency, f_o, gamma):
        """ Van Vleck-Weisskopf lineshape model

        :param f: Input frequencies in MHz (Size M) 
        :param fo: Line center frequencies in MHz (Size N)
        :param gamma: Half width at half maximum in MHz (Size NxO)

        Returns lineshape calculation (Size MxNxO)"""

        f = frequency
        fo = f_o[..., np.newaxis]
        gam = gamma[..., np.newaxis]

        shape = (1 / pi) * (f / fo)**2 * gam * (1 / ((fo - f)**2 + gam**2) + 1 / ((fo + f)**2 + gam**2))
        shape = np.moveaxis(shape, -1, 0)
        return shape

    @staticmethod
    def gross(frequency, f_o, gamma):
        """ Gross lineshape model

        :param f: Input frequencies in MHz (Size M) 
        :param fo: Line center frequencies in MHz (Size N)
        :param gamma: Half width at half maximum in MHz (Size NxO)

        Returns lineshape calculation (Size MxNxO)"""

        f = frequency
        fo = f_o[..., np.newaxis]
        gam = gamma[..., np.newaxis]

        shape = 1 / pi * (f / fo) * (4 * f * gam * fo) / ((fo**2 - f**2)**2 + 4 * f**2 * gam**2)
        shape = np.moveaxis(shape, -1, 0)
        return shape 

    @staticmethod
    def benreuven(frequency, f_o, gamma, zeta, delta):
        """ Ben Reuven lineshape model

        :param f: Input frequencies in MHz (Size M) 
        :param fo: Line center frequencies in MHz (Size N)
        :param gamma: Half width at half maximum in MHz (Size NxO)
        :param zeta: Line coupling coefficient (Size NxO)
        :param delta: Line shift coefficient in MHz (Size NxO)

        Returns lineshape calculation (Size MxNxO)"""

        f = frequency
        fo = f_o[..., np.newaxis]
        gam = gamma[..., np.newaxis]
        zet = zeta[..., np.newaxis]
        delt = delta[..., np.newaxis]

        shape = 2 / pi * (f / fo)**2 * ((gam - zet) * f**2 + (gam + zet) * ((fo + delt)**2 + gam**2 - zet**2)) / ((f**2 - (fo + delt)**2 - gam**2 + zet**2)**2 + 4 * f**2 * gam**2)
        shape = np.moveaxis(shape, -1, 0)
        return shape  

    @staticmethod
    def import_jpl_file(filename):
        """ Reads raw .cat files from the JPL Millimeter Spectral Line Catalog and convert them to a .mat file format
            https://spec.jpl.nasa.gov/ 
            Format is described in catalog documentation
            Last updated: May 2020"""

        line_file = open(filename, 'r')
        lines = line_file.readlines()
        line_number = len(lines)

        freq = np.zeros((line_number, 1))
        err = np.zeros((line_number, 1))
        lgint = np.zeros((line_number, 1))
        dr = np.zeros((line_number, 1))
        elo = np.zeros((line_number, 1))
        gup = np.zeros((line_number, 1))
        tag = np.zeros((line_number, 1))
        qnfmt = np.zeros((line_number, 1))

        if lines[0][51:55].strip(' ') == '1404':
            n_upper = np.zeros((line_number, 1))
            km_upper = np.zeros((line_number, 1))
            kp_upper = np.zeros((line_number, 1))
            v_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            km_lower = np.zeros((line_number, 1))
            kp_lower = np.zeros((line_number, 1))
            v_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'KM_UPPER', 'KP_UPPER', 'V_UPPER', 'N_LOWER', 'KM_LOWER', 'KP_LOWER', 'V_LOWER']
        elif lines[0][51:55].strip(' ') == '303': 
            n_upper = np.zeros((line_number, 1))
            km_upper = np.zeros((line_number, 1))
            kp_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            km_lower = np.zeros((line_number, 1))
            kp_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'KM_UPPER', 'KP_UPPER', 'N_LOWER', 'KM_LOWER', 'KP_LOWER']
        elif lines[0][51:55].strip(' ') == '1303': 
            n_upper = np.zeros((line_number, 1))
            k_upper = np.zeros((line_number, 1))
            v_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            k_lower = np.zeros((line_number, 1))
            v_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'K_UPPER', 'V_UPPER', 'N_LOWER', 'K_LOWER', 'V_LOWER']
        elif lines[0][51:55].strip(' ') == '202':  
            n_upper = np.zeros((line_number, 1))
            lam_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            lam_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'LAM_UPPER', 'N_LOWER', 'LAM_LOWER']
        elif lines[0][51:55].strip(' ') == '101':
            n_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'N_LOWER']
        elif lines[0][51:55].strip(' ') == '112':
            n_upper = np.zeros((line_number, 1))
            j_upper = np.zeros((line_number, 1))
            n_lower = np.zeros((line_number, 1))
            j_lower = np.zeros((line_number, 1))
            labels = ['FREQ (MHZ)', 'ERR (MHz)', 'LGINT (nm^2MHz)', 'DR', 'ELO (cm^-1)', 'GUP', 'TAG', 'QNFMT', 'N_UPPER', 'J_UPPER', 'N_LOWER', 'J_LOWER']
        else: 
            raise ValueError('Either the input file is not formatted properly or this type of catalog file is not currently supported for import')

        for index in range(0, line_number):
            freq[index] = float(lines[index][0:13])
            err[index] = float(lines[index][13:21])
            lgint[index] = float(lines[index][21:29])
            dr[index] = int(lines[index][29:31])
            elo[index] = float(lines[index][31:41])
            gup[index] = int(lines[index][41:44])
            tag[index] = int(lines[index][44:51])
            qnfmt[index] = int(lines[index][51:55])

            if lines[index][51:55].strip(' -') == '1404':
                n_upper[index] = int(lines[index][55:57])
                if 'a' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('a', ''))
                elif 'b' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('b', ''))
                elif 'c' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('c', ''))
                else:
                    km_upper[index] = int(lines[index][57:59])
                kp_upper[index] = int(lines[index][59:61])
                v_upper[index] = int(lines[index][61:63])
                n_lower[index] = int(lines[index][63:69])
                if 'a' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('a', ''))
                elif 'b' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('b', ''))
                elif 'c' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('c', ''))
                else:
                    km_lower[index] = int(lines[index][69:71])
                kp_lower[index] = int(lines[index][71:73])
                v_lower[index] = int(lines[index][73:75])
            elif lines[index][51:55].strip(' -') == '303':
                n_upper[index] = int(lines[index][55:57])
                if 'a' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('a', ''))
                elif 'b' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('b', ''))
                elif 'c' in lines[index][57:59]: 
                    km_upper[index] = int(lines[index][57:59].replace('c', ''))
                else:
                    km_upper[index] = int(lines[index][57:59])
                kp_upper[index] = int(lines[index][59:61])
                n_lower[index] = int(lines[index][61:69])
                if 'a' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('a', ''))
                elif 'b' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('b', ''))
                elif 'c' in lines[index][69:71]: 
                    km_lower[index] = int(lines[index][69:71].replace('c', ''))
                else:
                    km_lower[index] = int(lines[index][69:71])
                kp_lower[index] = int(lines[index][71:73])
            elif lines[index][51:55].strip(' -') == '1303':
                n_upper[index] = int(lines[index][55:57])
                if 'a' in lines[index][57:59]: 
                    k_upper[index] = int(lines[index][57:59].replace('a', ''))
                elif 'b' in lines[index][57:59]: 
                    k_upper[index] = int(lines[index][57:59].replace('b', ''))
                elif 'c' in lines[index][57:59]: 
                    k_upper[index] = int(lines[index][57:59].replace('c', ''))
                else:
                    k_upper[index] = int(lines[index][57:59])
                v_upper[index] = int(lines[index][59:61])
                n_lower[index] = int(lines[index][61:69])
                if 'a' in lines[index][69:71]: 
                    k_lower[index] = int(lines[index][69:71].replace('a', ''))
                elif 'b' in lines[index][69:71]: 
                    k_lower[index] = int(lines[index][69:71].replace('b', ''))
                elif 'c' in lines[index][69:71]: 
                    k_lower[index] = int(lines[index][69:71].replace('c', ''))
                else:
                    k_lower[index] = int(lines[index][69:71])
                v_lower[index] = int(lines[index][71:73])
            elif lines[index][51:55].strip(' -') == '202':
                n_upper[index] = int(lines[index][55:57])
                # Removing Hund's case indicators, as they aren't relevant here
                if 'a' in lines[index][57:59]: 
                    lam_upper[index] = int(lines[index][57:59].replace('a', ''))
                elif 'b' in lines[index][57:59]: 
                    lam_upper[index] = int(lines[index][57:59].replace('b', ''))
                elif 'c' in lines[index][57:59]: 
                    lam_upper[index] = int(lines[index][57:59].replace('c', ''))
                else:
                    lam_upper[index] = int(lines[index][57:59]) 
                n_lower[index] = int(lines[index][59:69])
                if 'a' in lines[index][69:71]: 
                    lam_lower[index] = int(lines[index][69:71].replace('a', ''))
                elif 'b' in lines[index][69:71]: 
                    lam_lower[index] = int(lines[index][69:71].replace('b', ''))
                elif 'c' in lines[index][69:71]: 
                    lam_lower[index] = int(lines[index][69:71].replace('c', ''))
                else:
                    lam_lower[index] = int(lines[index][69:71])
            elif lines[index][51:55].strip(' -') == '101': 
                n_upper[index] = int(lines[index][55:57])
                n_lower[index] = int(lines[index][57:69])
            elif lines[index][51:55].strip(' -') == '112':
                n_upper[index] = int(lines[index][55:57])
                # Removing Hund's case indicators, as they aren't relevant here
                if 'a' in lines[index][57:59]: 
                    j_upper[index] = int(lines[index][57:59].replace('a', ''))
                elif 'b' in lines[index][57:59]: 
                    j_upper[index] = int(lines[index][57:59].replace('b', ''))
                elif 'c' in lines[index][57:59]: 
                    j_upper[index] = int(lines[index][57:59].replace('c', ''))
                else:
                    j_upper[index] = int(lines[index][57:59]) 
                n_lower[index] = int(lines[index][59:69])
                if 'a' in lines[index][69:71]: 
                    j_lower[index] = int(lines[index][69:71].replace('a', ''))
                elif 'b' in lines[index][69:71]: 
                    j_lower[index] = int(lines[index][69:71].replace('b', ''))
                elif 'c' in lines[index][69:71]: 
                    j_lower[index] = int(lines[index][69:71].replace('c', ''))
                else:
                    j_lower[index] = int(lines[index][69:71])

        if lines[0][51:55].strip(' ') == '1404': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'KM_UPPER': km_upper, 'KP_UPPER': kp_upper, 'V_UPPER': v_upper, 'N_LOWER': n_lower, 'KM_LOWER': km_lower, 'KP_LOWER': kp_lower, 'V_LOWER': v_lower}
        elif lines[0][51:55].strip(' ') == '303': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'KM_UPPER': km_upper, 'KP_UPPER': kp_upper, 'N_LOWER': n_lower, 'KM_LOWER': km_lower, 'KP_LOWER': kp_lower}
        elif lines[0][51:55].strip(' ') == '1303': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'K_UPPER': k_upper, 'V_UPPER': v_upper, 'N_LOWER': n_lower, 'K_LOWER': k_lower, 'V_LOWER': v_lower}
        elif lines[0][51:55].strip(' ') == '202': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'LAM_UPPER': lam_upper, 'N_LOWER': n_lower, 'LAM_LOWER': lam_lower}
        elif lines[0][51:55].strip(' ') == '101': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'N_LOWER': n_lower}
        elif lines[0][51:55].strip(' ') == '112': 
            mdict = {'labels': labels, 'FREQ': freq, 'ERR': err, 'LGINT': lgint, 'DR': dr, 'ELO': elo, 'GUP': gup, 'QNFMT': qnfmt, 'N_UPPER': n_upper, 'J_UPPER': j_upper, 'N_LOWER': n_lower, 'J_LOWER': j_lower}

        sio.savemat(filename[:-4] + '.mat', mdict)
        line_file.close()


class PH3(molecule):
    """ PH3 opacity model from Hoffman, Steffes, and DeBoer 2001 """

    lineshape_type = 'Van Vleck-Weisskopf'
    linear = False 
    molar_mass = 33.99758  # g/mol
    triple_point = 139.41

    def __init__(self, **kwargs): 

        # Spectral line data 
        self.lines = sio.loadmat(os.path.join(self.jpl_path, 'c034003_HSD.mat'))
        self.line_labels = [x.strip(' ') for x in list(self.lines['labels'])]
        self.parameters = sio.loadmat(os.path.join(self.lineshape_path, 'PH3.mat'))
        self.parameter_labels = [x.strip(' ') for x in list(self.parameters['labels'])]

        super().__init__(**kwargs)

    def write_line_parameters(self): 
        """ Helper function for writing line-by-line width parameters to a .mat file
            This function also scales the spectral line intensities as discussed in Hoffman et al. 2001
        """

        lines = sio.loadmat(os.path.join(self.jpl_path, 'c034003.mat'))
        labels = ['GAMMA', 'GAMMA_TEMP']
        length = max(np.shape(lines['FREQ']))
        ph3 = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))

        inelastic_mask = ((lines['N_UPPER'] - lines['N_LOWER']) != 0).flatten()
        elastic_mask = ((lines['N_UPPER'] - lines['N_LOWER']) == 0).flatten()
        three_mask = (abs(lines['LAM_UPPER']) == 3).flatten() & (abs(lines['LAM_LOWER']) == 3).flatten() & elastic_mask
        six_mask = (abs(lines['LAM_UPPER']) == 6).flatten() & (abs(lines['LAM_LOWER']) == 6).flatten() & elastic_mask
        group1 = (six_mask | three_mask) & (abs(lines['N_UPPER']) < 8).flatten()
        group2 = three_mask & ((abs(lines['N_UPPER']) >= 8).flatten() & (abs(lines['N_UPPER']) <= 26).flatten())
        group3 = ((~(group1 | group2)) & elastic_mask)

        ph3[inelastic_mask, 0] = 4.2157 * (1e3 * spc.torr / spc.bar)  # Converting to MHz/torr from GHz/bar
        ph3[inelastic_mask, 1] = 1
        h2[inelastic_mask, 0] = 3.2930 * (1e3 * spc.torr / spc.bar)
        h2[inelastic_mask, 1] = 0.75
        he[inelastic_mask, 0] = 1.6803 * (1e3 * spc.torr / spc.bar)
        he[inelastic_mask, 1] = 0.75

        ph3[group3, 0] = 4.2157 * (1e3 * spc.torr / spc.bar)
        ph3[group3, 1] = 1
        h2[group3, 0] = 3.2930 * (1e3 * spc.torr / spc.bar)
        h2[group3, 1] = 0.75
        he[group3, 0] = 1.6803 * (1e3 * spc.torr / spc.bar)
        he[group3, 1] = 0.75

        ph3[group1, 0] = 0.4976 * (1e3 * spc.torr / spc.bar)
        ph3[group1, 1] = 1
        h2[group1, 0] = 1.4121 * (1e3 * spc.torr / spc.bar)
        h2[group1, 1] = 0.75
        he[group1, 0] = 0.7205 * (1e3 * spc.torr / spc.bar)
        he[group1, 1] = 0.75

        ph3[group2, 0] = 3.1723 * (1e3 * spc.torr / spc.bar)
        ph3[group2, 1] = 1
        h2[group2, 0] = 0.5978 * (1e3 * spc.torr / spc.bar)
        h2[group2, 1] = 0.75
        he[group2, 0] = 0.3050 * (1e3 * spc.torr / spc.bar)
        he[group2, 1] = 0.75

        mdict = {'labels': labels, 'PH3': ph3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'PH3.mat'), mdict)

        lines['LGINT'][group1] += log10(2.76)
        lines['LGINT'][group2] += log10(36.65)
        sio.savemat(os.path.join(self.jpl_path, '034003_HSD.mat'), lines)



class H2S(molecule): 
    """ H2S opacity model from DeBoer and Steffes 1994 """

    lineshape_type = 'Ben-Reuven'
    linear = False
    molar_mass = 34.1  # g/mol 
    molar_heat_capacity = 4.01 * spc.R  # J/mol K
    triple_point = 187.61

    def __init__(self, **kwargs): 

        # Spectral line data 
        self.lines = sio.loadmat(os.path.join(self.jpl_path, 'c034002.mat'))
        self.line_labels = [x.strip(' ') for x in list(self.lines['labels'])]
        self.parameters = sio.loadmat(os.path.join(self.lineshape_path, 'H2S.mat'))
        self.parameter_labels = [x.strip(' ') for x in list(self.parameters['labels'])]

        super().__init__(**kwargs)

    def write_line_parameters(self): 
        """ Helper function for writing line-by-line width parameters to a .mat file
            Laboratory measurements of some individual linewidths are included
        """ 

        lines = sio.loadmat(os.path.join(self.jpl_path, 'c034002.mat'))
        labels = ['GAMMA', 'GAMMA_TEMP', 'ZETA', 'ZETA_TEMP', 'DELTA']
        length = max(np.shape(lines['FREQ']))
        h2s = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))

        h2s[:, 0] = 5.78 * (1e3 * spc.torr / spc.bar)
        h2s[:, 1] = 0.7
        h2s[:, 2] = 5.78 * (1e3 * spc.torr / spc.bar)  
        h2s[:, 3] = 0.7
        h2s[:, 4] = 1.28 * (1e3 * spc.torr / spc.bar) 

        h2[:, 0] = 1.96 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.7
        h2[:, 2] = 1.96 * (1e3 * spc.torr / spc.bar)  
        h2[:, 3] = 0.7
        h2[:, 4] = 0

        he[:, 0] = 1.20 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 0.7
        he[:, 2] = 1.20 * (1e3 * spc.torr / spc.bar)  
        he[:, 3] = 0.7
        he[:, 4] = 0

        h2s[np.floor(lines['FREQ'] / 1e3).flatten() == 168, 0] = 5.38 * (1e3 * spc.torr / spc.bar)
        h2s[np.floor(lines['FREQ'] / 1e3).flatten() == 216, 0] = 6.82 * (1e3 * spc.torr / spc.bar)
        h2s[np.floor(lines['FREQ'] / 1e3).flatten() == 300, 0] = 5.82 * (1e3 * spc.torr / spc.bar)
        h2s[np.floor(lines['FREQ'] / 1e3).flatten() == 393, 0] = 5.08 * (1e3 * spc.torr / spc.bar)

        mdict = {'labels': labels, 'H2S': h2s, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'H2S.mat'), mdict)



class H2O(molecule): 
    """ H2O opacity models broadened by H2, He, N2
        Options for model parameter are 'KarpowiczSteffes', 'BellottiSteffes', and 'HodgesModified'
        Line models: 
        Based on calculations tabulated in Bauer et al. 1989 and selected laboratory measurements
        Continuum models: 
        For H2 and He: Karpowicz and Steffes 2011; Bellotti, Steffes, and Chinsomboon 2016; Hodges modifications
        Hodges modifications are empirical scalings to the Karpowicz and Steffes model opacity to obtain agreement with 
        600 MHz Juno measurements of the deep atmosphere of Jupiter

        The refractive index comes from the expression of Rueger 2002
    """

    lineshape_type = 'Van Vleck-Weisskopf'
    linear = False 
    molar_mass = 18.01528  # g/mol
    molar_heat_capacity = 4 * spc.R  # J/mol K
    triple_point = 273.16

    def __init__(self, model='BellottiSteffes', use_KS_lines=True, **kwargs): 
        self.model = model

        # Spectral linedata 
        if use_KS_lines: 
            self.lines = sio.loadmat(os.path.join(self.jpl_path, 'c018003_KS.mat'))
            self.parameters = sio.loadmat(os.path.join(self.lineshape_path, 'H2O_KS.mat'))
        else: 
            self.lines = sio.loadmat(os.path.join(self.jpl_path, 'c018003.mat'))
            self.parameters = sio.loadmat(os.path.join(self.lineshape_path, 'H2O.mat'))
        
        self.line_labels = [x.strip(' ') for x in list(self.lines['labels'])]    
        self.parameter_labels = [x.strip(' ') for x in list(self.parameters['labels'])]

        super().__init__(**kwargs)

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'):
        """ Calculates spectral line and continuum opacity of H2O vapor (see class description)

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')

            Returns the absorption of the gas (Size MxN)
        """ 

        line_absorption = super().absorption(frequency, gases, gases_index, temperature, pressure, units=units)

        gas_pressure = gases * pressure

        f = frequency
        gp_H2O = gas_pressure[gases_index.index('H2O')][..., np.newaxis]
        t = temperature[..., np.newaxis]

        if self.model == 'KarpowiczSteffes':
            # From Karpowicz and Steffes 2011 
            continuum_absorption = 4.36510480961e-12 * (f / 1e3)**2 * (300 / t)**13.3619799812 * (gp_H2O * 1e3)**2 + 2.10003048186e-31 * (f / 1e3)**2 * (300 / t)**0.0435525417274 * (gp_H2O * 1e3)**6.76418487001
        elif self.model == 'BellottiSteffes':
            # From Bellotti et al. 2016 
            continuum_absorption = 3.1e-12 * (f / 1e3)**2 * (300 / t)**12 * (gp_H2O * 1e3)**2
        elif self.model == 'HodgesModified':
            continuum_absorption = 4.36510480961e-12 * (f / 1e3)**2 * (300 / t)**13.3619799812 * (gp_H2O * 1e3)**2 + 2.10003048186e-32 * (f / 1e3)**2 * (300 / t)**0.0435525417274 * (gp_H2O * 1e3)**6.76418487001
        else: 
            raise ValueError('Specify continuum model as "KarpowiczSteffes", "BellottiSteffes", or "HodgesModified"')
        
        if 'N2' in gases_index: 
            # From Bauer et al. 1995 measurements at 239 GHz
            gp_N2 = gas_pressure[gases_index.index('N2')][..., np.newaxis]
            continuum_absorption += 0.6705e-8 * (f / 1e3)**2 / 239**2 * (296 / t)**(4.86) * (gp_N2 * spc.bar / spc.torr / 750) * (gp_H2O * spc.bar / spc.torr)
        if 'H2' in gases_index: 
            # From Karpowicz and Steffes 2011 (converted to invcm)
            gp_H2 = gas_pressure[gases_index.index('H2')][..., np.newaxis]
            continuum_absorption += 5.07722009423e-16 * (f / 1e3)**2 * (300 / t)**3 * (gp_H2 * 1e3) * (gp_H2O * 1e3)
        if 'He' in gases_index:
            # From Karpowicz and Steffes 2011 (converted to invcm)
            gp_He = gas_pressure[gases_index.index('He')][..., np.newaxis]
            continuum_absorption += 1.03562010226e-15 * (f / 1e3)**2 * (300 / t)**3 * (gp_He * 1e3) * (gp_H2O * 1e3)

        continuum_absorption = np.moveaxis(continuum_absorption, -1, 0)
        if units == 'dBperkm': 
            continuum_absorption *= self.OpticaldepthstodBperkm
        elif units == 'invcm': 
            pass

        absorption = line_absorption + continuum_absorption

        if self.model == 'HodgesModified':
            # Empiricial scaling factor for high pressures/temperatures
            scale = np.zeros(len(pressure))
            P_top = 150 
            P_bot = 2000
            f_bot = 16 
            scale[pressure < P_top] = 0
            scale[pressure > P_bot] = 1
            mid_pressure = pressure[(pressure > P_top) & (pressure < P_bot)]
            scale[(pressure > P_top) & (pressure < P_bot)] = (mid_pressure - P_top) / (P_bot - P_top)
            scale_factor = (1 - scale) + scale * f_bot
            absorption = absorption * scale_factor

        return absorption

    def refractivity(self, temperature, pressure): 
        """ Calculates refractivity  of H2O vapor (see class description)

            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of water (Size N)

            :return: Refractivity in ppm 
        """
        mbar = pressure * 1e3
        N = 71.97 * mbar / temperature + 3.75406e5 * mbar / temperature**2
        return N

    def write_line_parameters(self):
        """ Helper function for writing line-by-line width parameters to a .mat file
            Tables of lines and linewidths can be found in Karpowicz and Steffes 2011 and Bauer et al. 1989
        """ 
        lines = sio.loadmat(os.path.join(self.jpl_path, 'c018003.mat'))
        labels = ['GAMMA', 'GAMMA_TEMP']
        length = max(np.shape(lines['FREQ']))
        h2o = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))
        co2 = np.zeros((length, 2))
        n2 = np.zeros((length, 2)) 
        o2 = np.zeros((length, 2))

        # From Karpowicz and Steffes 2011
        ks_frequency = np.array([22.2351, 183.3101, 321.2256, 325.1529, 380.1974, 439.1508, 443.0183, 448.0011, 470.8890, 474.6891, 488.4911, 556.9360, 620.7008, 752.0332, 916.1712])
        ks_intens = np.array([0.1314E-13, 0.2279E-11, 0.8058E-13, 0.2701E-11, 0.2444E-10, 0.2185E-11, 0.4637E-12, 0.2568E-10, 0.8392E-12, 0.3272E-11, 0.6676E-12, 0.1535E-8, 0.1711E-10, 0.1014e-8, 0.4238E-10])
        ks_intens = np.log10(ks_intens * 1e8)
        ks_eo = np.array([2.144, 0.668, 6.179, 1.541, 1.048, 3.595, 5.048, 1.405, 3.597, 2.379, 2.852, 0.159, 2.391, 0.396, 1.441]) / (self.hc / spc.k / self.To)
        ks_h2o_gamma = np.array([13.49, 14.66, 10.57, 13.81, 14.54, 9.715, 7.88, 12.75, 9.83, 10.95, 13.13, 14.05, 11.836, 12.53, 12.75]) * (1e3 * spc.torr / spc.bar)
        ks_h2o_gamma_temp = np.array([0.61, 0.85, 0.54, 0.74, 0.89, 0.62, 0.5, 0.67, 0.65, 0.64, 0.72, 1, 0.68, 0.84, 0.78])
        ks_h2_gamma = np.array([2.395, 2.4, 2.395, 2.395, 2.39, 2.395, 2.395, 2.395, 2.395, 2.395, 2.395, 2.395, 2.395, 2.395, 2.395]) * (1e3 * spc.torr / spc.bar)
        ks_h2_gamma_temp = np.array([0.9, 0.95, 0.9, 0.9, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        ks_he_gamma = np.array([0.67, 0.71, 0.67, 0.67, 0.63, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67, 0.67])
        ks_he_gamma_temp = np.array([0.515, 0.490, 0.515, 0.490, 0.540, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515])

        # From Bauer et al. 1989
        bauer_frequency = np.array([22.2351, 71.5924, 139.6143, 183.3101, 321.2256, 325.1529, 339.0440, 380.1974, 390.1345, 437.3467, 439.1508, 443.0183, 448.0011, 458.6828, 470.8890, 474.6891, 488.4911, 503.5685, 504.4827, 530.3429, 534.2405, 556.9360, 571.9137, 591.6934, 593.2278, 620.7008, 645.7661, 645.9057, 752.0332, 766.7936, 826.5499, 841.0507, 854.0498, 863.8392, 863.8600, 906.2059, 916.1712, 960.6547, 968.0306, 970.3150, 987.9268])
        bauer_h2o_gamma = np.array([17.70, 18.80, 18.80, 18.80, 14.40, 18, 17, 18.80, 13.80, 10.40, 12, 10.50, 17, 17, 13.10, 14.60, 17.50, 8.55, 8.62, 12.30, 16.80, 17.60, 10.50, 13.90, 15.40, 15.20, 7.30, 7.30, 16.70, 11.10, 17.60, 12.20, 10.90, 6.74, 6.74, 15.10, 17, 16.20, 16.10, 16.80, 18.10])
        bauer_h2o_gamma_temp = np.array([0.61, 0.82, 0.79, 0.79, 0.54, 0.74, 0.61, 0.82, 0.55, 0.48, 0.52, 0.5, 0.67, 0.78, 0.65, 0.64, 0.72, 0.43, 0.45, 0.48, 0.67, 1, 0.43, 0.67, 0.66, 0.68, 0.43, 0.43, 0.84, 0.36, 1, 0.45, 0.43, 0.47, 0.47, 0.53, 0.78, 0.77, 0.70, 0.67, 0.9])
        bauer_co2_gamma = np.array([5.75, 6.08, 6.24, 6.24, 4.07, 6.06, 5.75, 6.08, 3.86, 3.96, 4.11, 3.99, 4.92, 6.11, 4.21, 4.38, 5.20, 3.84, 3.84, 3.42, 4.64, 7.95, 2.89, 3.89, 4.0, 4.95, 3.71, 3.71, 7.79, 3.13, 7.95, 3.23, 3.05, 3.57, 3.57, 4.44, 6.11, 4.79, 4.75, 4.64, 7.62])
        bauer_co2_gamma_temp = np.array([0.52, 0.6, 0.57, 0.57, 0.46, 0.61, 0.52, 0.6, 0.53, 0.74, 0.72, 0.75, 0.56, 0.61, 0.68, 0.65, 0.43, 0.76, 0.76, 0.67, 0.53, 0.65, 0.77, 0.61, 0.65, 0.55, 0.77, 0.77, 0.68, 0.81, 0.65, 0.76, 0.84, 0.78, 0.78, 0.42, 0.61, 0.43, 0.43, 0.53, 0.65])
        bauer_n2_gamma = np.array([3.87, 4.11, 4.23, 4.23, 3.33, 3.99, 3.87, 4.11, 3.11, 2.64, 3.01, 2.66, 3.78, 3.82, 3.09, 3.38, 3.74, 2.31, 2.31, 2.72, 3.68, 4.59, 2.04, 3.09, 3.27, 3.49, 2.05, 2.05, 4.38, 2.08, 4.59, 2.34, 1.87, 1.84, 1.84, 3.48, 3.82, 3.55, 3.54, 3.68, 4.26])
        bauer_n2_gamma_temp = np.array([0.69, 0.69, 0.7, 0.7, 0.67, 0.68, 0.69, 0.69, 0.62, 0.60, 0.62, 0.60, 0.65, 0.70, 0.65, 0.64, 0.68, 0.61, 0.61, 0.48, 0.63, 0.69, 0.17, 0.60, 0.57, 0.71, 0.65, 0.65, 0.68, 0.28, 0.69, 0.33, 0.23, 0.69, 0.69, 0.70, 0.70, 0.71, 0.69, 0.63, 0.69])
        bauer_o2_gamma = np.array([2.47, 2.63, 2.72, 2.72, 2.05, 2.57, 2.47, 2.63, 1.89, 1.72, 1.95, 1.74, 2.42, 2.51, 2.00, 2.18, 2.37, 1.52, 1.52, 1.55, 2.28, 3.04, 1.09, 1.88, 2.02, 2.28, 1.33, 1.33, 2.89, 1.11, 3.04, 1.26, 0.99, 1.17, 1.17, 2.14, 2.51, 2.20, 2.19, 2.28, 2.85])
        bauer_o2_gamma_temp = np.array([0.71, 0.70, 0.72, 0.72, 0.70, 0.69, 0.71, 0.70, 0.66, 0.61, 0.67, 0.62, 0.73, 0.70, 0.69, 0.72, 0.72, 0.58, 0.58, 0.51, 0.70, 0.67, 0.23, 0.61, 0.63, 0.74, 0.56, 0.56, 0.65, 0.30, 0.67, 0.34, 0.26, 0.56, 0.56, 0.71, 0.70, 0.70, 0.71, 0.70, 0.65])

        h2o[:, 0] = np.mean(bauer_h2o_gamma)
        h2o[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 0] = bauer_h2o_gamma
        h2o[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 0] = ks_h2o_gamma
        h2o[:, 1] = np.mean(bauer_h2o_gamma_temp)
        h2o[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 1] = bauer_h2o_gamma_temp
        h2o[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 1] = ks_h2o_gamma_temp

        h2[:, 0] = np.mean(ks_h2_gamma)
        h2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 0] = ks_h2_gamma
        h2[:, 1] = np.mean(ks_h2_gamma_temp)
        h2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 1] = ks_h2_gamma_temp

        he[:, 0] = np.mean(ks_he_gamma)
        he[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 0] = ks_he_gamma
        he[:, 1] = np.mean(ks_he_gamma_temp)
        he[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), ks_frequency), 1] = ks_he_gamma_temp

        n2[:, 0] = np.mean(bauer_n2_gamma)
        n2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 0] = bauer_n2_gamma
        n2[:, 1] = np.mean(bauer_n2_gamma_temp)
        n2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 1] = bauer_n2_gamma_temp

        o2[:, 0] = np.mean(bauer_o2_gamma)
        o2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 0] = bauer_o2_gamma
        o2[:, 1] = np.mean(bauer_o2_gamma_temp)
        o2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 1] = bauer_o2_gamma_temp

        co2[:, 0] = np.mean(bauer_co2_gamma)
        co2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 0] = bauer_co2_gamma
        co2[:, 1] = np.mean(bauer_co2_gamma_temp)
        co2[np.searchsorted(np.round(lines['FREQ'].flatten() / 1e3, 4), bauer_frequency), 1] = bauer_co2_gamma_temp

        mdict = {'labels': labels, 'H2O': h2o, 'H2': h2, 'He': he, 'N2': n2, 'O2': o2, 'CO2': co2}
        sio.savemat(os.path.join(self.lineshape_path, 'H2O.mat'), mdict)

        # Adding a second file for lines only Karpowicz used to make his water model
        lines['FREQ'] = ks_frequency.reshape(len(ks_frequency), 1) * 1e3
        lines['LGINT'] = ks_intens.reshape(len(ks_frequency), 1)
        lines['ELO'] = ks_eo.reshape(len(ks_frequency), 1)
        sio.savemat(self.path.join(self.jpl_path, 'c018003_KS.mat'), lines)

        labels = ['GAMMA', 'GAMMA_TEMP']
        length = max(np.shape(ks_intens))
        h2o = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))
        h2o[:, 0] = ks_h2o_gamma.flatten()
        h2o[:, 1] = ks_h2o_gamma_temp.flatten()
        h2[:, 0] = ks_h2_gamma.flatten()
        h2[:, 1] = ks_h2_gamma_temp.flatten()
        he[:, 0] = ks_he_gamma.flatten()
        he[:, 1] = ks_he_gamma_temp.flatten()
        mdict = {'labels': labels, 'H2O': h2o, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'H2O_KS.mat'), mdict)


class NH3(molecule): 
    """ NH3 opacity models broadened by H2, He, H2O, and CH4
        Options for model parameter are 'HanleySteffes', 'DevarajSteffes', 'BellottiSteffes', and 'HodgesModified'
        The Hanley and Steffes 2009 model uses only the NH3 inversion lines 
        The Devaraj and Steffes 2011 model uses inversion, rotovibrational, and rotational lines. A pressure-dependent switch is incorporated for 
        the inversion line broadening parameters
        The Belotti and Steffes 2016 model uses inversion, rotovibrational, and rotational lines. A frequency-dependent switch is incorporated for 
        the inversion line broadening parameters
        Hodges modifications are empirical scalings to the Hanley and Steffes model to obtain agreement with 
        600 MHz Juno measurements of the deep atmosphere of Jupiter
    """

    molar_mass = 17.031  # g/mol
    molar_heat_capacity = 4.46 * spc.R  # J/mol K
    linear = False
    triple_point = 195.5

    def __init__(self, model='BellottiSteffes', **kwargs):
        self.model = model 
        super().__init__(**kwargs)

        if self.model == 'HanleySteffes' or self.model == 'HodgesModified': 
            self.inv_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_HS.mat'))
            self.inv_parameters = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_inversion_HS.mat'))
            self.inv_parameter_labels = [x.strip(' ') for x in list(self.inv_parameters['labels'])]
        elif self.model == 'DevarajSteffes':
            self.inv_lines_lp = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_DS_lowpressure.mat'))
            self.inv_parameters_lp = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_inversion_DS_lowpressure.mat'))
            self.inv_parameter_lp_labels = [x.strip(' ') for x in list(self.inv_parameters_lp['labels'])]
            self.inv_lines_hp = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_DS_highpressure.mat'))
            self.inv_parameters_hp = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_inversion_DS_highpressure.mat'))
            self.inv_parameter_hp_labels = [x.strip(' ') for x in list(self.inv_parameters_hp['labels'])]
            self.rot_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_rotational_DS.mat'))
            self.rot_parameters = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_rotational_DS.mat'))
            self.rot_parameter_labels = [x.strip(' ') for x in list(self.rot_parameters['labels'])]
            self.vib_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_rotovibrational_DS.mat'))
            self.vib_parameters = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_rotovibrational_DS.mat'))
            self.vib_parameter_labels = [x.strip(' ') for x in list(self.vib_parameters['labels'])]
        elif self.model == 'BellottiSteffes':
            self.inv_lines_lf = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_BS_lowfrequency.mat'))
            self.inv_parameters_lf = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_inversion_BS_lowfrequency.mat'))
            self.inv_parameter_lf_labels = [x.strip(' ') for x in list(self.inv_parameters_lf['labels'])]
            self.inv_lines_hf = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_BS_highfrequency.mat'))
            self.inv_parameters_hf = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_inversion_BS_highfrequency.mat'))
            self.inv_parameter_hf_labels = [x.strip(' ') for x in list(self.inv_parameters_hf['labels'])]
            self.rot_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_rotational_BS.mat'))
            self.rot_parameters = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_rotational_BS.mat'))
            self.rot_parameter_labels = [x.strip(' ') for x in list(self.rot_parameters['labels'])]
            self.vib_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3_rotovibrational_BS.mat'))
            self.vib_parameters = sio.loadmat(os.path.join(self.lineshape_path, 'NH3_rotovibrational_BS.mat'))
            self.vib_parameter_labels = [x.strip(' ') for x in list(self.vib_parameters['labels'])]
        else: 
            raise ValueError('Specify model as "HanleySteffes", "DevarajSteffes", "BellottiSteffes", or "HodgesModified"')

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'): 
        """ Calculates spectral line opacity of NH3 (see class description)

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')

            Returns the absorption of the gas (Size MxN)
        """ 

        absorption = np.zeros((len(frequency), len(pressure)))
        gas_pressure = gases * pressure * spc.bar / spc.torr

        if self.model == 'HanleySteffes' or self.model == 'HodgesModified':
            gamma, zeta, delta = self.load_line_parameters(temperature, gas_pressure, gases_index, self.inv_parameters, self.inv_parameter_labels)
            alpha_max = self.absorption_coeff(10**self.inv_lines['LGINT'].ravel(), temperature, self.inv_lines['ELO'].ravel(), 
                                              gases[np.array(gases_index) == self.__class__.__name__, :] * pressure * spc.bar / spc.torr, gamma)
            inv_lineshape = self.benreuven(frequency, self.inv_lines['FREQ'].ravel(), gamma, zeta, delta)
            absorption = np.sum(alpha_max * spc.pi * gamma * inv_lineshape, axis=-1) 

        elif self.model == 'DevarajSteffes': 
            lp_mask = pressure <= 15
            hp_mask = pressure >= 15 
            lp_mat_mask = np.zeros((len(frequency), len(pressure)), dtype=bool) + lp_mask
            hp_mat_mask = np.zeros((len(frequency), len(pressure)), dtype=bool) + hp_mask 
            low_pressure = pressure[lp_mask]
            high_pressure = pressure[hp_mask]
            low_temperature = temperature[lp_mask]
            high_temperature = temperature[hp_mask]
            low_gas_pressure = gas_pressure[:, lp_mask]
            high_gas_pressure = gas_pressure[:, hp_mask]

            if low_pressure.size > 0:
                gamma_lp, zeta_lp, delta_lp = self.load_line_parameters(low_temperature, low_gas_pressure, gases_index, self.inv_parameters_lp, self.inv_parameter_lp_labels)
                alpha_max_lp = self.absorption_coeff(10**self.inv_lines_lp['LGINT'].ravel(), low_temperature, self.inv_lines_lp['ELO'].ravel(), 
                                                     low_gas_pressure[gases_index.index('NH3')], gamma_lp)
                inv_lp_lineshape = self.benreuven(frequency, self.inv_lines_lp['FREQ'].ravel(), gamma_lp, zeta_lp, delta_lp)
                absorption_lp = np.sum(alpha_max_lp * spc.pi * gamma_lp * inv_lp_lineshape, axis=-1) 
                np.place(absorption, lp_mat_mask, absorption_lp)
            if high_pressure.size > 0:
                gamma_hp, zeta_hp, delta_hp = self.load_line_parameters(high_temperature, high_gas_pressure, gases_index, self.inv_parameters_hp, self.inv_parameter_hp_labels)
                alpha_max_hp = self.absorption_coeff(10**self.inv_lines_hp['LGINT'].ravel(), high_temperature, self.inv_lines_hp['ELO'].ravel(), 
                                                     high_gas_pressure[gases_index.index('NH3')], gamma_hp)
                inv_hp_lineshape = self.benreuven(frequency, self.inv_lines_hp['FREQ'].ravel(), gamma_hp, zeta_hp, delta_hp)
                absorption_hp = np.sum(alpha_max_hp * spc.pi * gamma_hp * inv_hp_lineshape, axis=-1) 
                np.place(absorption, hp_mat_mask, absorption_hp)

            gamma_rot = self.load_line_parameters(temperature, gas_pressure, gases_index, self.rot_parameters, self.rot_parameter_labels)
            gamma_vib = self.load_line_parameters(temperature, gas_pressure, gases_index, self.vib_parameters, self.vib_parameter_labels)
            alpha_max_rot = self.absorption_coeff(10**self.rot_lines['LGINT'].ravel(), temperature, self.rot_lines['ELO'].ravel(), 
                                                  gases[gases_index.index('NH3')] * pressure * spc.bar / spc.torr, gamma_rot)
            alpha_max_vib = self.absorption_coeff(10**self.vib_lines['LGINT'].ravel(), temperature, self.vib_lines['ELO'].ravel(),
                                                  gases[gases_index.index('NH3')] * pressure * spc.bar / spc.torr, gamma_vib)

            rot_lineshape = self.gross(frequency, self.rot_lines['FREQ'].ravel(), gamma_rot)
            vib_lineshape = self.gross(frequency, self.vib_lines['FREQ'].ravel(), gamma_vib)
            absorption += np.sum(alpha_max_rot * spc.pi * gamma_rot * rot_lineshape, axis=-1) 
            absorption += np.sum(alpha_max_vib * spc.pi * gamma_vib * vib_lineshape, axis=-1) 

        elif self.model == 'BellottiSteffes': 
            lf_mask = frequency <= 30e3
            hf_mask = frequency > 30e3
            lf_mat_mask = np.transpose(np.zeros((len(pressure), len(frequency)), dtype=bool) + lf_mask)
            hf_mat_mask = np.transpose(np.zeros((len(pressure), len(frequency)), dtype=bool) + hf_mask)
            low_frequency = frequency[lf_mask]
            high_frequency = frequency[hf_mask]

            if low_frequency.size > 0:
                gamma_lf, zeta_lf, delta_lf = self.load_line_parameters(temperature, gas_pressure, gases_index, self.inv_parameters_lf, self.inv_parameter_lf_labels)
                alpha_max_lf = self.absorption_coeff(10**self.inv_lines_lf['LGINT'].ravel(), temperature, self.inv_lines_lf['ELO'].ravel(),
                                                     gas_pressure[gases_index.index('NH3')], gamma_lf)
                inv_lf_lineshape = self.benreuven(low_frequency, self.inv_lines_lf['FREQ'].ravel(), gamma_lf, zeta_lf, delta_lf)
                absorption_lf = np.sum(alpha_max_lf * spc.pi * gamma_lf * inv_lf_lineshape, axis=-1) 
                np.place(absorption, lf_mat_mask, absorption_lf)
            if high_frequency.size > 0:
                gamma_hf, zeta_hf, delta_hf = self.load_line_parameters(temperature, gas_pressure, gases_index, self.inv_parameters_hf, self.inv_parameter_hf_labels)
                alpha_max_hf = self.absorption_coeff(10**self.inv_lines_hf['LGINT'].ravel(), temperature, self.inv_lines_hf['ELO'].ravel(), 
                                                     gas_pressure[gases_index.index('NH3')], gamma_hf)
                inv_hf_lineshape = self.benreuven(high_frequency, self.inv_lines_hf['FREQ'].ravel(), gamma_hf, zeta_hf, delta_hf)
                absorption_hf = np.sum(alpha_max_hf * spc.pi * gamma_hf * inv_hf_lineshape, axis=-1) 
                np.place(absorption, hf_mat_mask, absorption_hf)

            gamma_rot = self.load_line_parameters(temperature, gas_pressure, gases_index, self.rot_parameters, self.rot_parameter_labels)
            gamma_vib = self.load_line_parameters(temperature, gas_pressure, gases_index, self.vib_parameters, self.vib_parameter_labels)
            alpha_max_rot = self.absorption_coeff(10**self.rot_lines['LGINT'].ravel(), temperature, self.rot_lines['ELO'].ravel(), 
                                                  gases[gases_index.index('NH3')] * pressure * spc.bar / spc.torr, gamma_rot)
            alpha_max_vib = self.absorption_coeff(10**self.vib_lines['LGINT'].ravel(), temperature, self.vib_lines['ELO'].ravel(), 
                                                  gases[gases_index.index('NH3')] * pressure * spc.bar / spc.torr, gamma_vib)

            rot_lineshape = self.gross(frequency, self.rot_lines['FREQ'].ravel(), gamma_rot)
            vib_lineshape = self.gross(frequency, self.vib_lines['FREQ'].ravel(), gamma_vib)

            absorption += np.sum(alpha_max_rot * spc.pi * gamma_rot * rot_lineshape, axis=-1) 
            absorption += np.sum(alpha_max_vib * spc.pi * gamma_vib * vib_lineshape, axis=-1) 

        if self.model == 'HodgesModified':
            scale_factor = np.zeros(len(pressure))
            P_top = 150 
            P_bot = 4000
            f_bot = 0.8 
            scale_factor[pressure < P_top] = 1
            scale_factor[pressure > P_bot] = f_bot
            mid_pressure = pressure[(pressure > P_top) & (pressure < P_bot)]
            scale = (mid_pressure - P_top) / (P_bot - P_top)
            scale_factor[(pressure > P_top) & (pressure < P_bot)] = (1 - scale) + scale * f_bot
            absorption = absorption * scale_factor

        if units == 'dBperkm': 
            absorption *= self.OpticaldepthstodBperkm
        elif units == 'invcm': 
            pass
        else:
            raise ValueError('Specify valid units (invcm or dBperkm')
        return absorption

    def load_line_parameters(self, temperature, pressure, gases_index, parameters, parameter_labels):
        """ Calculates linewidths from parameter tables and atmospheric conditions. 

            :param temperature: Temperature in Kelvins of the gas mixture (Size M)
            :param pressure: Matrix of gas pressures in torr (Size NxM)
            :param gases_index: row indexing for the pressure array Size(N)
            :param parameters: Ben-Reuven shape parameters as loaded from mat files 
            :param parameter_labels: Column index for shape parameters

            Returns line parameters (Size OxN) where O is the number of spectral lines for the molecule
            This is overwritten because NH3 specifically has several separately maintained spectral line and linewidth parameter files 
        """
        index = [x for x in gases_index if x in parameters.keys()]
        if self.model == 'HanleySteffes' or self.model == 'HodgesModified':
            gamma = 0
            zeta = 0
            delta = 0
            for i in index: 
                if i == 'NH3':
                    self.To = 295
                    gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                    zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                    self.To = 300
                else:  
                    gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                    zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                delta = parameters[i][:, parameter_labels.index('DELTA')] * gamma

            return gamma, zeta, delta 
        elif self.model == 'DevarajSteffes':
            if parameters is self.inv_parameters_lp or parameters is self.inv_parameters_hp: 
                gamma = 0
                zeta = 0
                delta = 0
                for i in index: 
                    if i == 'NH3':
                        self.To = 295
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                        self.To = 300
                    else:  
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                    delta = parameters[i][:, parameter_labels.index('DELTA')] * gamma

                return gamma, zeta, delta
            elif parameters is self.rot_parameters or parameters is self.vib_parameters: 
                gamma = 0
                for i in index: 
                    if i == 'NH3':
                        self.To = 295
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        self.To = 300
                    else:  
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])

                return gamma

        elif self.model == 'BellottiSteffes':
            if (parameters is self.inv_parameters_lf) or (parameters is self.inv_parameters_hf): 
                gamma = 0
                zeta = 0
                delta = 0
                for i in index: 
                    if i == 'NH3':
                        self.To = 295
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                        self.To = 300
                    else:  
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        zeta += self.linewidth_func(parameters[i][:, parameter_labels.index('ZETA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('ZETA_TEMP')])
                    delta = parameters[i][:, parameter_labels.index('DELTA')] * gamma

                return gamma, zeta, delta
            elif parameters is self.rot_parameters or parameters is self.vib_parameters: 
                gamma = 0
                for i in index: 
                    if i == 'NH3':
                        self.To = 295
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])
                        self.To = 300
                    else:  
                        gamma += self.linewidth_func(parameters[i][:, parameter_labels.index('GAMMA')], pressure[gases_index.index(i), :], self.To / temperature, parameters[i][:, parameter_labels.index('GAMMA_TEMP')])

                return gamma

    def write_line_parameters(self):
        """ Helper function for writing line-by-line width parameters to a .mat file
        """ 

        # Hanley and Steffes model
        # Inversion lines
        inv_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'nh3lincat190.mat'))
        freq = inv_lines['fo'] * 1e3
        lgint = log10(0.9301 * inv_lines['Io'] * 2.99792458e18)
        elo = inv_lines['Eo']
        j = inv_lines['J']
        k = inv_lines['K']

        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo, 'J': j, 'K': k}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_HS.mat'), mdict)

        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP', 'ZETA', 'ZETA_TEMP', 'DELTA']
        nh3 = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))

        nh3[:, 0] = 0.852 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1 
        nh3[:, 2] = 0.5296 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 3] = 1.554
        nh3[:, 4] = -0.0498

        h2[:, 0] = 1.640 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.7756
        h2[:, 2] = 1.262 * (1e3 * spc.torr / spc.bar)
        h2[:, 3] = 0.7964
        h2[:, 4] = -0.0498

        he[:, 0] = 0.75 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 2 / 3
        he[:, 2] = 0.3 * (1e3 * spc.torr / spc.bar)
        he[:, 3] = 2 / 3
        he[:, 4] = -0.0498

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_inversion_HS.mat'), mdict)

        # Devaraj and Steffes model (with pressure switch)
        # Inversion lines 
        inv_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_inversion.mat'))
        freq = inv_lines['fo'] * 1e3
        lgint = log10(0.9862 * inv_lines['Io'] * 2.99792458e18)
        elo = inv_lines['Eo']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_DS_lowpressure.mat'), mdict)
        lgint = log10(1.3746 * inv_lines['Io'] * 2.99792458e18)
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_DS_highpressure.mat'), mdict)

        # Low pressure (<15 bar)
        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP', 'ZETA', 'ZETA_TEMP', 'DELTA']
        nh3 = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))
        h2o = np.zeros((length, 5))

        nh3[:, 0] = 0.7298 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1 
        nh3[:, 2] = 0.5152 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 3] = 2 / 3
        nh3[:, 4] = -0.0627 

        h2[:, 0] = 1.7465 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.8202
        h2[:, 2] = 1.2163 * (1e3 * spc.torr / spc.bar)
        h2[:, 3] = 0.8873
        h2[:, 4] = -0.0627

        he[:, 0] = 0.9779 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 1
        he[:, 2] = 0.0291 * (1e3 * spc.torr / spc.bar)
        he[:, 3] = 0.8994
        he[:, 4] = -0.0627

        h2o[:, 0] = 8.4993 * (1e3 * spc.torr / spc.bar)
        h2o[:, 1] = 1
        h2o[:, 2] = 1.364 * (1e3 * spc.torr / spc.bar)
        h2o[:, 3] = 2 / 3
        h2o[:, 4] = -0.0627

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he, 'H2O': h2o}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_inversion_DS_lowpressure.mat'), mdict)

        # High pressure (>15 bars)
        nh3 = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))
        h2o = np.zeros((length, 5))

        nh3[:, 0] = 0.7298 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1 
        nh3[:, 2] = 0.5152 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 3] = 2 / 3
        nh3[:, 4] = 0.2 

        h2[:, 0] = 1.6361 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.8
        h2[:, 2] = 1.1313 * (1e3 * spc.torr / spc.bar)
        h2[:, 3] = 0.6234
        h2[:, 4] = 0.2

        he[:, 0] = 0.4555 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 0.5
        he[:, 2] = 0.1 * (1e3 * spc.torr / spc.bar)
        he[:, 3] = 0.5
        he[:, 4] = 0.2

        h2o[:, 0] = 8.4993 * (1e3 * spc.torr / spc.bar)
        h2o[:, 1] = 1
        h2o[:, 2] = 1.364 * (1e3 * spc.torr / spc.bar)
        h2o[:, 3] = 2 / 3
        h2o[:, 4] = 0.2

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he, 'H2O': h2o}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_inversion_DS_highpressure.mat'), mdict)

        # Rotational Lines
        rot_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_rotational.mat'))
        freq = rot_lines['fo_rot'] * 1e3
        lgint = log10(2.4268 * rot_lines['Io_rot'] * 2.99792458e18)
        elo = rot_lines['Eo_rot']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_rotational_DS.mat'), mdict)

        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP']
        nh3 = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))

        nh3[:, 0] = 3.1789 * rot_lines['gNH3_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1
        h2[:, 0] = 0.2984 * rot_lines['gH2_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.8730
        he[:, 0] = 0.75 * rot_lines['gHe_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 2 / 3

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_rotational_DS.mat'), mdict)

        # Rotovibrational lines

        vib_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_rotovibrational.mat'))
        freq = vib_lines['fo_v2'] * 1e3
        lgint = log10(1.1206 * vib_lines['Io_v2'] * 2.99792458e18)
        elo = vib_lines['Eo_v2']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_rotovibrational_DS.mat'), mdict)

        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP']
        nh3 = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))

        nh3[:, 0] = 9.5 * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1
        h2[:, 0] = 1.4 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.73
        he[:, 0] = 0.68 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 0.5716

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_rotovibrational_DS.mat'), mdict)

        # Bellotti and Steffes model (with frequency switch)
        # Inversion lines
        inv_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_inversion.mat'))
        freq = inv_lines['fo'] * 1e3
        lgint = log10(0.9619 * inv_lines['Io'] * 2.99792458e18)
        elo = inv_lines['Eo']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_BS_lowfrequency.mat'), mdict)
        lgint = log10(0.9862 * inv_lines['Io'] * 2.99792458e18)
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_inversion_BS_highfrequency.mat'), mdict)

        # Low frequency (<30 GHz)
        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP', 'ZETA', 'ZETA_TEMP', 'DELTA']
        nh3 = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))
        h2o = np.zeros((length, 5))
        ch4 = np.zeros((length, 5))

        nh3[:, 0] = 0.7523 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1 
        nh3[:, 2] = 0.6162 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 3] = 1.3832
        nh3[:, 4] = -0.0139

        h2[:, 0] = 1.6937 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.8085
        h2[:, 2] = 1.3263 * (1e3 * spc.torr / spc.bar)
        h2[:, 3] = 0.8199
        h2[:, 4] = -0.0139

        he[:, 0] = 0.6997 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 1
        he[:, 2] = 0.1607 * (1e3 * spc.torr / spc.bar)
        he[:, 3] = 0
        he[:, 4] = -0.0139

        h2o[:, 0] = 5.3119 * (1e3 * spc.torr / spc.bar)
        h2o[:, 1] = 0.6224
        h2o[:, 2] = 5.2333 * (1e3 * spc.torr / spc.bar)
        h2o[:, 3] = 2.1248
        h2o[:, 4] = -0.0139

        ch4[:, 0] = 2.6406 * (1e3 * spc.torr / spc.bar)
        ch4[:, 1] = 1
        ch4[:, 2] = 0.9111 * (1e3 * spc.torr / spc.bar)
        ch4[:, 3] = 1.92
        ch4[:, 4] = -0.0139

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he, 'H2O': h2o, 'CH4': ch4}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_inversion_BS_lowfrequency.mat'), mdict)

        # High frequency (>30 GHz)
        nh3 = np.zeros((length, 5))
        h2 = np.zeros((length, 5))
        he = np.zeros((length, 5))
        h2o = np.zeros((length, 5))
        ch4 = np.zeros((length, 5))

        nh3[:, 0] = 0.7298 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1 
        nh3[:, 2] = 0.5152 * inv_lines['gammaNH3o'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 3] = 2 / 3
        nh3[:, 4] = -0.0627

        h2[:, 0] = 1.7465 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.8202
        h2[:, 2] = 1.2163 * (1e3 * spc.torr / spc.bar)
        h2[:, 3] = 0.8873
        h2[:, 4] = -0.0627

        he[:, 0] = 0.9979 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 1
        he[:, 2] = 0.0291 * (1e3 * spc.torr / spc.bar)
        he[:, 3] = 0.8994
        he[:, 4] = -0.0627

        h2o[:, 0] = 5.3119 * (1e3 * spc.torr / spc.bar)
        h2o[:, 1] = 0.6224
        h2o[:, 2] = 5.2333 * (1e3 * spc.torr / spc.bar)
        h2o[:, 3] = 2.1248
        h2o[:, 4] = -0.0627

        ch4[:, 0] = 2.6406 * (1e3 * spc.torr / spc.bar)
        ch4[:, 1] = 1
        ch4[:, 2] = 0.9111 * (1e3 * spc.torr / spc.bar)
        ch4[:, 3] = 1.92
        ch4[:, 4] = -0.0627

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he, 'H2O': h2o, 'CH4': ch4}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_inversion_BS_highfrequency.mat'), mdict)

        # Rotational Lines
        rot_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_rotational.mat'))
        freq = rot_lines['fo_rot'] * 1e3
        lgint = log10(2.7252 * rot_lines['Io_rot'] * 2.99792458e18)
        elo = rot_lines['Eo_rot']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_rotational_BS.mat'), mdict)

        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP']
        nh3 = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))

        nh3[:, 0] = 3.1518 * rot_lines['gNH3_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 1
        h2[:, 0] = 1.7761 * rot_lines['gH2_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.5
        he[:, 0] = 0.6175 * rot_lines['gHe_rot'].flatten() * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 0.5663

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_rotational_BS.mat'), mdict)

        # Rotovibrational lines

        vib_lines = sio.loadmat(os.path.join(self.line_path, 'nh3', 'ammonia_rotovibrational.mat'))
        freq = vib_lines['fo_v2'] * 1e3
        lgint = log10(0.7286 * vib_lines['Io_v2'] * 2.99792458e18)
        elo = vib_lines['Eo_v2']
        mdict = {'FREQ': freq, 'LGINT': lgint, 'ELO': elo}
        sio.savemat(os.path.join(self.line_path, 'nh3', 'nh3_rotovibrational_BS.mat'), mdict)

        length = len(freq)
        labels = ['GAMMA', 'GAMMA_TEMP']
        nh3 = np.zeros((length, 2))
        h2 = np.zeros((length, 2))
        he = np.zeros((length, 2))

        nh3[:, 0] = 5.0894 * (1e3 * spc.torr / spc.bar)
        nh3[:, 1] = 0.9996
        h2[:, 0] = 0.5982 * (1e3 * spc.torr / spc.bar)
        h2[:, 1] = 0.5
        he[:, 0] = 0.6175 * (1e3 * spc.torr / spc.bar)
        he[:, 1] = 0.5505

        mdict = {'labels': labels, 'NH3': nh3, 'H2': h2, 'He': he}
        sio.savemat(os.path.join(self.lineshape_path, 'NH3_rotovibrational_BS.mat'), mdict)

    
# Collisionally induced opacities 

class H2(molecule):
    """ Model for opacity of H2 in combination with He and CH4 from Orton 2006 
        Specified as absorption per amagat squared for equilibrium (1:1) para vs ortho ratios of hydrogen 
        as well as normal (1:3) para vs ortho ratios. 
        Refractive index is taken from measurements of Essen 1953
        Note: There are other absorption tables from HITRAN that give absorption per number density squared for several para/ortho ratios
    """

    molar_heat_capacity = 3.5 * spc.R  # J/mol K, depends on ortho-para rate, but this is the high temperature approximation
    
    triple_point = 13.81  # K
    molar_mass = 2.01588  # g/mol 

    def __init__(self, para_ortho='equilibrium', **kwargs): 
        self.para_ortho = para_ortho
        self.cia = sio.loadmat(os.path.join(self.cia_path, 'orton_files', 'H2_absorption.mat'))
        super().__init__(**kwargs)

        # RectBivariateSpline is faster method but requires inputs to be sorted
        # Temperature interpolation in log space may be more accurate
        if self.para_ortho == 'equilibrium': 
            self.H2H2 = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2H2_equilibrium'])
            self.H2He = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2He_equilibrium'])
            self.H2CH4 = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2CH4_equilibrium'])
            # Molar heat capacity equilibrium expression of Farkas
            temps = np.array([0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329])
            a = np.array([2.5, 2.5014, 2.6333, 2.9628, 3.4459, 4.2345, 4.5655, 3.8721, 3.3806, 3.2115, 3.1946, 3.2402, 3.3035, 3.3630, 3.411, 3.4439, 3.5])
            self.molar_heat_capacity = spi.interp1d(temps, a * spc.R, kind='cubic', bounds_error=False, fill_value=(2.5 * spc.R, 3.5 * spc.R))
        elif self.para_ortho == 'normal': 
            self.H2H2 = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2H2_normal'])
            self.H2He = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2He_normal'])
            self.H2CH4 = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['H2CH4_normal'])
            # Molar heat capacity frozen equilibrium expression of Trafton (close to the normal curve)
            temps = np.array([0, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 273.1, 329])
            a = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5022, 2.5154, 2.6369, 2.8138, 2.9708, 3.0976, 3.2037, 3.2899, 3.3577, 3.4085, 3.4424, 3.5])
            self.molar_heat_capacity = spi.interp1d(temps, a * spc.R, kind='cubic', bounds_error=False, fill_value=(2.5 * spc.R, 3.5 * spc.R))

        else: 
            raise ValueError('Specify either "normal" or "equilibrium" para-ortho ratios')

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'): 
        """ Calculates the collisional absorption for H2 mixtures by interpolating over a grid

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')

        """

        # For this routine and for other collisional absorption tables 
        # There's a bit of axis juggling that needs to be done 
        # to make sure that broadcasting works correctly

        temperature = temperature[..., np.newaxis]
        gas_pressure = gases[..., np.newaxis] * pressure[..., np.newaxis]
        wavenumber = frequency * 1e6 / (spc.c * 1e2)
        number_density = gas_pressure / (spc.R * 10 * temperature) * spc.N_A
        absorption = number_density[gases_index.index('H2')]**2 * exp(self.H2H2.ev(wavenumber, temperature)) / self.L**2

        if 'He' in gases_index: 
            absorption += number_density[gases_index.index('H2')] * number_density[gases_index.index('He')] * exp(self.H2He.ev(wavenumber, temperature)) / self.L**2
        if 'CH4' in gases_index: 
            absorption += number_density[gases_index.index('H2')] * number_density[gases_index.index('CH4')] * exp(self.H2CH4.ev(wavenumber, temperature)) / self.L**2
        # Assume a frequency squared dependence for all values below the lowest recorded wavenumber 
        mask = wavenumber.ravel() < self.cia['wavenumber'][:, 0]
        absorption[..., mask] = absorption[..., mask] * wavenumber[mask]**2 / self.cia['wavenumber'][:, 0]**2 
        absorption = np.moveaxis(absorption, -1, 0)

        if units == 'dBperkm': 
            absorption *= self.OpticaldepthstodBperkm
        elif units == 'invcm': 
            pass
        else:
            raise ValueError('Specify valid units (invcm or dBperkm')

        return absorption

    def refractivity(self, temperature, pressure): 
        """ Calculates refractivity of H2

            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)

            :return: Refractivity in ppm 

            Essen gives 136, and Newell and Baird (via Hanley's dissertation) gives the value below
        """
        N = 135.77 * pressure / (spc.atm / spc.bar) * spc.zero_Celsius / temperature
        return N


class He(molecule): 
    """ Refractive index data for He comes from Essen 1953
        Microwave absorption of He-He collisions assumed to be zero 
    """ 

    molar_heat_capacity = 2.5 * spc.R  # J/mol K
    triple_point = 1.76  # K 
    molar_mass = 4.0026  # g/mol

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'): 
        """ Placeholder for He absorption, since any microwave opacity comes from collisions with
            other molecules and are accounted for there. Returns appropriately sized matrix of zeros

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')

        """
        return np.zeros((len(frequency), *np.shape(pressure)))

    def refractivity(self, temperature, pressure): 
        """ Calculates refractivity of He

            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :return: Refractivity in ppm 
            Essen gives 35, and Newell and Baird (via Hanley's dissertation) gives the value below

        """
        N = 34.51 * pressure / (spc.atm / spc.bar) * spc.zero_Celsius / temperature
        return N


class CH4(molecule):
    """ Model for opacity of CH4 from Orton 2006 
        Specified as absorption per amagat squared  
        Refractive index is from Spilker's Ph.D. dissertation 1990
        Note: He collisional contributions are not currently taken into account 
    """

    molar_heat_capacity = 4.5 * spc.R  # J/mol K
    triple_point = 90.7  # Kelvin
    molar_mass = 16.04  # g/mol

    def __init__(self, **kwargs):
        self.cia = sio.loadmat(os.path.join(self.cia_path, 'orton_files', 'CH4_absorption.mat'))
        super().__init__(**kwargs)
        self.CH4CH4 = spi.RectBivariateSpline(self.cia['wavenumber'].flatten(), self.cia['temperature'].flatten(), self.cia['CH4CH4'])

    def absorption(self, frequency, gases, gases_index, temperature, pressure, units='dBperkm'): 
        """ Calculates the collisional absorption for CH4 mixtures by interpolating over a grid

            :param frequency: Frequency in MHz (Size M)
            :param gases: Matrix of gas volume mole fractions (Size OxN)
            :param gases_index: row indexing for the gases array (Size O)
            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :param units: Output units ('dBperkm' or 'invcm')
        """

        temperature = temperature[..., np.newaxis]
        gas_pressure = gases[..., np.newaxis] * pressure[..., np.newaxis]
        wavenumber = frequency * 1e6 / (spc.c * 1e2)
        number_density = gas_pressure / (spc.R * 10 * temperature) * spc.N_A
        absorption = number_density[gases_index.index('CH4')]**2 * exp(self.CH4CH4.ev(wavenumber, temperature)) / self.L**2
        mask = wavenumber.ravel() < self.cia['wavenumber'][:, 0]
        absorption[..., mask] = absorption[..., mask] * wavenumber[mask]**2 / self.cia['wavenumber'][:, 0]**2 
        absorption = np.moveaxis(absorption, -1, 0)

        if units == 'dBperkm': 
            absorption *= self.OpticaldepthstodBperkm
        elif units == 'invcm': 
            pass
        else:
            raise ValueError('Specify valid units (invcm or dBperkm')

        return absorption

    def refractivity(self, temperature, pressure): 
        """ Calculates refractivity of CH4

            :param temperature: Temperature in Kelvins of the gas mixture (Size N)
            :param pressure: Pressure in bars of the gas mixture (Size N)
            :return: Refractivity in ppm 
        """
        N = 440 * pressure / (spc.atm / spc.bar) * spc.zero_Celsius / temperature
        return N 



