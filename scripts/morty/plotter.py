import matplotlib.pyplot as plt
import os.path
import os
import numpy as np

class PlotterCollection:
    def __init__(self, num_nucs, imdir='./images/'):
        """
        This class holds various plotters

        Parameters
        ----------
        run_params : dict
            key : str
                Name of run parameter
        data_params : dict
            ket : str
                Name of data parameter
        imdir : str
            Path to image directory

        """
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
        plt.rcParams['savefig.dpi'] = 300
        self.imdir = imdir
        self.num_nucs = num_nucs
        if not os.path.isdir(self.imdir):
            os.mkdir(self.imdir)
        return
    
    def plot_time(self, data_dict):
        """
        Plots average data over time where `data_dict` contains all
         necessary information

        Parameters
        ----------
        data_dict : dict
            key : str
                Variable name
        """
        for nuclide_i in range(self.num_nucs):
            for i, x in enumerate(data_dict['xs']):
                y = data_dict[f'spat_avg_y_method{i}_nuc{nuclide_i}']
                lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']
                plt.plot(x, y, label=lab)
            plt.xlabel(data_dict['xlab'])
            plt.ylabel(data_dict['ylab'])
            plt.legend()
            plt.savefig(f'{self.imdir}{data_dict["savename"]}_{nuclide_i}.png')
            plt.close()
        for nuclide_i in range(self.num_nucs):
            for i, x in enumerate(data_dict['xs']):
                y = data_dict[f'spat_avg_y_method{i}_nuc{nuclide_i}']
                lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']
                plt.plot(x, y, label=lab)
        plt.xlabel(data_dict['xlab'])
        plt.ylabel(data_dict['ylab'])
        plt.legend()
        plt.savefig(f'{self.imdir}{data_dict["savename"]}_allnucs.png')
        plt.close()
        return
