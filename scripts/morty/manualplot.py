import matplotlib.pyplot as plt
import numpy as np

class PlotHolder:

    def __init__(self):
        self.data = {}
        self.times = {}
        self.msre = 'MSRE Data'
        self.adder = 'MCNP/ADDER'

        self.data[self.msre] = {}
        self.data[self.adder] = {}
        return
    
    def collect_data(self, cur_data=None, label=''):
        self._add_MSRE_data()
        self._add_ADDER_data()
        self._add_cur_data(cur_data, label)
        return
    
    def plot_data(self, imdir):
        #self._setup_print()
        nucs = self.data[self.adder].keys()
        colors = ['blue', 'orange', 'green', 'cyan', 'olive', 'pink']
        for nuc in nucs:
            other_count = 0
            for version in self.data.keys():
                if version == self.msre:
                    marker = '.'
                    linestyle = ''
                    color = 'black'
                elif version == self.adder:
                    marker = ''
                    linestyle = '-.'
                    color = 'red'
                else:
                    marker = ''
                    linestyle = '--'
                    color = colors[other_count%len(colors)]
                    other_count += 1
                try:
                    plt.plot(np.asarray(self.times[version])/(24*3600),
                            self.data[version][nuc], label=version,
                            linestyle=linestyle, marker=marker, color=color,
                            markersize=5)
                except KeyError:
                    continue
            plt.xlabel('Time [d]')
            plt.ylabel('Concentration [atoms/cc]')
            plt.legend(prop={'size': 10})
            plt.savefig(f'{imdir}experimental_{nuc}.png')
            plt.close()
        return

    
    def _setup_print(self):
        plt.rcParams["font.size"] = 16
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["lines.linewidth"] = 1.5
        plt.rcParams["lines.markersize"] = 1
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.which"] = "major"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 6.0
        plt.rcParams["ytick.major.size"] = 6.0
        plt.rcParams["xtick.minor.size"] = 3.0
        plt.rcParams["ytick.minor.size"] = 3.0
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['savefig.dpi'] = 600
        return

    
    def _add_MSRE_data(self):
        self.times[self.msre] = [
        1.028880E+07,
        1.046160E+07,
        1.331280E+07,
        1.409040E+07,
        1.469520E+07,
        2.221200E+07,
        2.921040E+07]

        self.data[self.msre]['Zr95'] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        2.04176E+16,
        1.85615E+16,
        1.67053E+16
        ]

        self.data[self.msre]['Nb95'] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        4.06E+16,
        8.11E+15,
        1.13E+15
        ]

        return
    
    def _add_ADDER_data(self):
        adder_times = [
        0.00E+00,
        4.17E-01,
        4.17E-01,
        1.00E+00,
        1.00E+00,
        1.50E+00,
        1.50E+00,
        2.00E+00,
        2.00E+00,
        3.17E+00,
        3.17E+00,
        7.70E+01,
        7.70E+01,
        8.00E+01,
        8.00E+01,
        8.10E+01,
        8.10E+01,
        8.23E+01,
        8.23E+01,
        8.40E+01,
        8.40E+01,
        8.60E+01,
        8.60E+01,
        8.70E+01,
        8.70E+01,
        8.75E+01,
        8.75E+01,
        8.80E+01,
        8.80E+01,
        8.85E+01,
        8.85E+01,
        9.10E+01,
        9.10E+01,
        9.50E+01,
        9.50E+01,
        1.05E+02,
        1.05E+02,
        1.08E+02,
        1.08E+02,
        1.09E+02,
        1.09E+02,
        1.12E+02,
        1.12E+02,
        1.14E+02,
        1.14E+02,
        1.15E+02,
        1.15E+02,
        1.17E+02,
        1.17E+02,
        1.23E+02,
        1.23E+02,
        1.40E+02,
        1.40E+02,
        1.41E+02,
        1.41E+02,
        1.42E+02,
        1.42E+02,
        1.54E+02,
        1.54E+02,
        1.58E+02,
        1.58E+02,
        1.64E+02,
        1.64E+02,
        1.71E+02,
        1.71E+02,
        1.72E+02,
        1.72E+02,
        1.74E+02,
        1.74E+02,
        2.62E+02,
        2.62E+02,
        2.65E+02,
        2.65E+02,
        2.65E+02,
        2.65E+02,
        2.80E+02,
        2.80E+02,
        2.88E+02,
        2.88E+02,
        2.93E+02,
        2.93E+02,
        2.94E+02,
        2.94E+02,
        2.95E+02,
        2.95E+02,
        2.96E+02,
        2.96E+02,
        2.97E+02,
        2.97E+02,
        3.25E+02,
        3.25E+02,
        3.39E+02,
        3.39E+02]

        self.times[self.adder] = np.asarray(adder_times) * 24 * 3600

        self.data[self.adder]['Zr95'] = [
        0.00E+00,
        8.70E+12,
        8.70E+12,
        8.86E+12,
        8.86E+12,
        2.98E+13,
        2.98E+13,
        3.00E+13,
        3.00E+13,
        1.28E+14,
        1.28E+14,
        5.81E+13,
        5.81E+13,
        3.09E+14,
        3.09E+14,
        3.06E+14,
        3.06E+14,
        5.66E+14,
        5.66E+14,
        5.58E+14,
        5.58E+14,
        9.67E+14,
        9.67E+14,
        9.59E+14,
        9.59E+14,
        1.16E+15,
        1.16E+15,
        1.16E+15,
        1.16E+15,
        1.36E+15,
        1.36E+15,
        1.33E+15,
        1.33E+15,
        2.95E+15,
        2.95E+15,
        2.65E+15,
        2.65E+15,
        3.90E+15,
        3.90E+15,
        3.89E+15,
        3.89E+15,
        5.65E+15,
        5.65E+15,
        5.55E+15,
        5.55E+15,
        6.00E+15,
        6.00E+15,
        5.90E+15,
        5.90E+15,
        8.70E+15,
        8.70E+15,
        7.20E+15,
        7.20E+15,
        7.74E+15,
        7.74E+15,
        7.71E+15,
        7.71E+15,
        1.38E+16,
        1.38E+16,
        1.32E+16,
        1.32E+16,
        1.59E+16,
        1.59E+16,
        1.89E+16,
        1.89E+16,
        1.87E+16,
        1.87E+16,
        1.94E+16,
        1.94E+16,
        7.47E+15,
        7.47E+15,
        8.45E+15,
        8.45E+15,
        8.43E+15,
        8.43E+15,
        1.42E+16,
        1.42E+16,
        1.30E+16,
        1.30E+16,
        1.52E+16,
        1.52E+16,
        1.50E+16,
        1.50E+16,
        1.55E+16,
        1.55E+16,
        1.53E+16,
        1.53E+16,
        1.58E+16,
        1.58E+16,
        1.17E+16,
        1.17E+16,
        1.71E+16,
        1.71E+16
        ]

        self.data[self.adder]['Nb95'] = [
            0.00E+00,
            1.91E+10,
            1.91E+10,
            7.41E+10,
            7.41E+10,
            1.76E+11,
            1.76E+11,
            3.35E+11,
            3.35E+11,
            1.31E+12,
            1.31E+12,
            3.42E+13,
            3.42E+13,
            3.80E+13,
            3.80E+13,
            4.06E+13,
            4.06E+13,
            4.54E+13,
            4.54E+13,
            5.42E+13,
            5.42E+13,
            6.82E+13,
            6.82E+13,
            7.72E+13,
            7.72E+13,
            8.21E+13,
            8.21E+13,
            8.75E+13,
            8.75E+13,
            9.34E+13,
            9.34E+13,
            1.24E+14,
            1.24E+14,
            2.04E+14,
            2.04E+14,
            4.41E+14,
            4.41E+14,
            5.19E+14,
            5.19E+14,
            5.35E+14,
            5.35E+14,
            6.74E+14,
            6.74E+14,
            7.55E+14,
            7.55E+14,
            8.13E+14,
            8.13E+14,
            8.96E+14,
            8.96E+14,
            1.23E+15,
            1.23E+15,
            2.13E+15,
            2.13E+15,
            2.17E+15,
            2.17E+15,
            2.19E+15,
            2.19E+15,
            2.99E+15,
            2.99E+15,
            3.34E+15,
            3.34E+15,
            3.83E+15,
            3.83E+15,
            4.57E+15,
            4.57E+15,
            4.68E+15,
            4.68E+15,
            4.96E+15,
            4.96E+15,
            5.79E+15,
            5.79E+15,
            5.72E+15,
            5.72E+15,
            5.71E+15,
            5.71E+15,
            5.86E+15,
            5.86E+15,
            6.08E+15,
            6.08E+15,
            6.24E+15,
            6.24E+15,
            6.28E+15,
            6.28E+15,
            6.32E+15,
            6.32E+15,
            6.37E+15,
            6.37E+15,
            6.42E+15,
            6.42E+15,
            6.81E+15,
            6.81E+15,
            7.09E+15,
            7.09E+15
        ]

    def _add_cur_data(self, cur_data, label):
        self.data[label] = {}
        self.times[label] = np.asarray(cur_data['times']) * 24 * 3600
        for nuc in cur_data['data'].keys():
            self.data[label][nuc] = cur_data['data'][nuc]
        return
        
