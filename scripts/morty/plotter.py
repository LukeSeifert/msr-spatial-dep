import matplotlib.pyplot as plt
import os.path
import os
import numpy as np
from time import time

class PlotterCollection:
    def __init__(self, imdir='./images/'):
        """
        This class holds various plotters

        Parameters
        ----------
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
        if not os.path.isdir(self.imdir):
            os.mkdir(self.imdir)
        return
    
    def parasitic_plot(self, parasitic_matrix, run_params, data_params):
        """
        Plots the parasitic absorption density

        Parameters
        ----------
        parasitic_abs_val : :class:`np.ndarray`
            Parasitic absorption for nuclides over time in
            atoms per cc
        
        """
        print(np.shape(parasitic_matrix) == ())
        input(f'{parasitic_matrix.keys()=}')

        if np.shape(parasitic_matrix) == ():
            input(f'{parasitic_matrix.keys()=}')
            pass
        else:
            for nuclide_index in range(np.shape(parasitic_matrix)[0]):
                x = run_params['times']
                y = parasitic_matrix[nuclide_index]
                lab = data_params['tracked_nucs'][nuclide_index]
                plt.plot(x, y, label=lab)
        plt.xlabel('Time [s]')
        plt.ylabel('Concentration [atoms/cc]')
        plt.savefig(f'{self.imdir}parasitic_absorption.png')
        plt.close()
        return
    
    def plot_time(self, data_dict, spatial_eval_node=None, y_scale='log'):
        """
        Plots average data over time where `data_dict` contains all
         necessary information

        Parameters
        ----------
        data_dict : dict
            key : str
                Variable name
        spatial_eval_node : int (optional)
            Spatial node index to evaluate    
        """
        num_nucs = []
        for i, x in enumerate(data_dict['xs']):
            num_nucs.append(np.shape(data_dict['ys'][i])[2])
        num_nucs = max(num_nucs)
        for nuclide_i in range(num_nucs):
            for i, x in enumerate(data_dict['xs']):
                if type(spatial_eval_node) == type(None):
                    try:
                        y = data_dict[f'spat_avg_y_method{i}_nuc{nuclide_i}']
                    except KeyError:
                        continue
                    ending = 'avg'
                else:
                    try:
                        y = data_dict['ys'][i][:, spatial_eval_node, nuclide_i]
                    except IndexError:
                        continue
                    ending = f'spat{spatial_eval_node}'
                lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']
                plt.plot(x, y, label=lab)
            plt.xlabel(data_dict['xlab'])
            plt.ylabel(data_dict['ylab'])
            plt.yscale(y_scale)
            plt.legend()
            try:
                plt.savefig(f'{self.imdir}{data_dict["savename"]}_{nuclide_i}_{ending}.png')
            except UnboundLocalError:
                print(f'No data for {nuclide_i} figure at spatial index {spatial_eval_node}')
            plt.close()
        return

    def gif_generate(self, data_dict, run_params):
        """
        Plots spatial conc over time where `data_dict` contains all
         necessary information

        Parameters
        ----------
        data_dict : dict
            key : str
                Variable name
        run_params : dict
            key : str
                Variable name
        """
        start_time = time()
        from matplotlib.animation import FuncAnimation
        plt.rcParams['savefig.dpi'] = 100
        fig, ax = plt.subplots()
        num_nucs = []
        for i, x in enumerate(data_dict['xs']):
            num_nucs.append(np.shape(data_dict['ys'][i])[2])
        num_nucs = max(num_nucs)
        max_conc = np.max(data_dict['ys'])

        def update(frame):
            ax.clear()
            plt.xlabel('Space [cm]')
            plt.vlines(run_params['core_outlet'], 0, 1e1 * max_conc, color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.ylim((1e-5 * max_conc, 1e1 * max_conc))
            plt.yscale('log')

            for nuclide_i in range(num_nucs):
                for i, x in enumerate(data_dict['xs']):
                    try:
                        y = data_dict['ys'][i][frame, :, nuclide_i]
                    except KeyError:
                        continue
                    lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']

                    ax.plot(run_params['positions'], y, label=lab, marker='.')
                    ax.set_title(f'Time: {round(frame*run_params["dt"], 4)} s')
                    plt.legend()
            

        animation = FuncAnimation(fig, update, frames=len(run_params['times']), interval=1)
        animation.save(f'{self.imdir}isobar_evolution.gif', writer='pillow')
        plt.close()
        plt.rcParams['savefig.dpi'] = 300
        print(f'Gif creation took: {round(time() - start_time, 3)} s')
        return


