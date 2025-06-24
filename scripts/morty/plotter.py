import matplotlib.pyplot as plt
import os.path
import os
import numpy as np
from time import time
import pandas as pd
from analysis import AnalysisCollection
from manualplot import PlotHolder
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

class PlotterCollection:
    def __init__(self, plotting_params, run_params, data_params):
        """
        This class holds various plotters

        Parameters
        ----------
        plotting_params : dict
            key : str
                Name of variable
        run_params : dict
            key : str
                Name of variable
        data_params : dict
            key : str
                Name of variable

        """
        plt.rcParams['agg.path.chunksize'] = 10000
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
        self.yscale = plotting_params['y_scale']
        plt.rcParams['savefig.dpi'] = 600
        self.imdir = plotting_params['image_directory']
        self.plotting_params = plotting_params
        self.run_params = run_params
        self.data_params = data_params
        if not os.path.isdir(self.imdir):
            os.mkdir(self.imdir)
        return

    def _time_lab(self):
        """
        Determines time scaling to use and labelling

        Returns
        -------
        scaling_factor : float
            Factor to scale time in (s) by
        xlab : str
            Time label to use
        """
        tf = self.run_params['final_time']
        second = 1
        millisecond = 1e-3 * second
        minute = 60 * second
        hour = 60 * minute
        day = 24 * hour
        year = 365.25 * day
        time_list = [day, hour, minute, second, millisecond]
        label_list = ['d', 'hr', 'min', 's', 'ms']
        for ti, time in enumerate(time_list):
            if time > tf:
                continue
            xlab = f'Time [{label_list[ti]}]'
            scaling_factor = 1 / time
            return scaling_factor, xlab
    
    def parasitic_plot(self, parasitic_data):
        """
        Plots the parasitic absorption density

        Parameters
        ----------
        parasitic_data : dict
            key : str
                Name of variable
        
        """
        parasitic_use_data = parasitic_data['parasitic']
        scale_fac, xlab = self._time_lab()
        for nuclide_index in range(np.shape(parasitic_use_data)[1]):
            for method_i in range(len(parasitic_data['parasitic'])):
                method_name = parasitic_data['labs'][method_i]
                x = self.run_params['times'] * scale_fac
                y = parasitic_use_data[method_i][nuclide_index]
                lab = self.data_params['tracked_nucs'][nuclide_index]
                plt.plot(x, y, label=f'{method_name} {lab} Captures')
            plt.yscale(self.yscale)
            plt.xlabel(xlab)
            plt.ylabel('Absorptions [atoms/cc]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.imdir}parasitic_absorption_{nuclide_index}.png')
            plt.close()
        
        for nuclide_index in range(np.shape(parasitic_use_data)[1]):
            for method_i in range(len(parasitic_data['parasitic'])):
                method_name = parasitic_data['labs'][method_i]
                x = self.run_params['times'] * scale_fac
                if method_i == 0:
                    y_base = parasitic_use_data[method_i][nuclide_index]
                else:
                    y = parasitic_use_data[method_i][nuclide_index]
                lab = self.data_params['tracked_nucs'][nuclide_index]
                if method_i != 0:
                    pcnt_diff = ((y_base - y) / (y_base) * 100) 
                    plt.plot(x, pcnt_diff, label=f'{method_name} {lab} Captures')
            plt.yscale(self.yscale)
            plt.xlabel(xlab)
            plt.ylabel('Difference [%]')
            plt.tight_layout()
            plt.savefig(f'{self.imdir}parasitic_absorption_pcntdiff_{nuclide_index}.png')
            plt.close()
        return
    
    def _time_plot_helper(self, data_dict, nuclide_i, ending,
                          spatial_eval_node, num_plot, manual_y_lab=False,
                          legend=True):
        plt.xlabel(data_dict['xlab'])
        plt.yscale(self.yscale)
        if not manual_y_lab:
            plt.ylabel(data_dict['ylab'])
        else:
            plt.yscale('log')
            #plt.ylabel(manual_y_lab, fontsize=14)
            plt.ylabel(manual_y_lab)
        if num_plot > 1 and legend:
            plt.legend()
        try:
            plot_main_name = f'{self.imdir}{data_dict["savename"]}'
            plt.savefig(f'{plot_main_name}_{nuclide_i}_{ending}.png')
        except UnboundLocalError:
            pos = spatial_eval_node
            print(f'No data for {nuclide_i} figure at spatial index {pos}')
        except OverflowError:
            print(f'Not plotting {plot_main_name} {nuclide_i} {ending}')
        plt.close()
        return
    
    def _data_collector(self, data_dict, data_str, spatial_eval_node,
                        ending, nuclide_i, print_diffs, pcnt_diff_bool=False,
                        print_data=False):
        initial_data = True
        num_plot = 0
        legend_opt = True
        if type(spatial_eval_node) == type(None):
            for i, x in enumerate(data_dict['xs']):
                try:
                    lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']
                    y = data_dict[f'{data_str}_avg_y_method{i}_nuc{nuclide_i}']
                    nuc = self.data_params['tracked_nucs'][nuclide_i]
                    
                    if data_str == 'spat':
                        self.plot_dict['times'] = data_dict['xs'][i]
                        self.plot_dict['data'][nuc] = data_dict[f'spat_avg_y_method{i}_nuc{nuclide_i}']
                        self.plot_obj.collect_data(self.plot_dict, data_dict['labs'][i])

                except KeyError:
                    continue
                ending = ending
                plt.plot(x, y, label=lab)
                num_plot += 1
                if print_data:
                    with open('data.txt', 'a') as f:
                        f.write(f'{lab = }\n')
                        f.write(f'{x = }\n')
                        f.write(f'{y = }\n\n')
                if initial_data:
                    base_y = y
                    base_lab = lab
                    initial_data = False
                if i != 0 and print_diffs:
                    pcnt_diff = (y[-1] - base_y[-1]) / (y[-1]) * 100
                    print(f'% diff {data_str} {lab}-{base_lab}: {pcnt_diff:.3E}%')
                if print_diffs:
                    print(f'    Final val {lab}: {y[-1]:.3E}')
            manual_y_lab = False
            if data_str == 'gradient':
                #manual_y_lab = r'(N$_{max}$ - N$_{min}$)/N$_{avg}$'
                manual_y_lab = r'Relative Gradient'
                legend_opt = False
            elif data_str == 'std':
                manual_y_lab = r'Standard Deviation'
                legend_opt = False
            self._time_plot_helper(data_dict, nuclide_i, ending,
                                    spatial_eval_node, num_plot,
                                    manual_y_lab=manual_y_lab,
                                    legend=legend_opt)
            if pcnt_diff_bool and not initial_data:
                legend_opt = False
                for i, x in enumerate(data_dict['xs']):
                    if i == 0:
                        continue
                    lab = f'{base_lab} - {data_dict[f"lab_method{i}_nuc{nuclide_i}"]}'
                    lab = 'Difference Metric'
                    y = data_dict[f'{data_str}_avg_y_method{i}_nuc{nuclide_i}']
                    pcnt_diffs = ((base_y - y) / (base_y) * 100)
                    pcnt_diffs[np.isnan(pcnt_diffs)] = 0
                    print(f'Peak Diff: {max(np.abs(pcnt_diffs)) = }')
                    print(f'Final Diff: {pcnt_diffs[-1] = }')
                    legend_opt = False
                    plt.plot(x, pcnt_diffs, label=lab)
                    num_plot += 1
                plt.xlabel(data_dict['xlab'])
                try:
                    min_y = np.min(pcnt_diffs)
                except UnboundLocalError:
                    return
                mindex = np.where(pcnt_diffs == min_y)[0][0]
                x_min = x[mindex]
                plt.plot(x[-1], pcnt_diffs[-1], marker='*', markersize=8,
                        color='orange', label='Equilibrium', linestyle='')
                plt.plot(x_min, min_y,
                         marker='s', markersize=8, color='blue',
                         label='Peak', linestyle='')
                plt.legend()
                plt.ylabel('Difference [%]')
                plt.yscale(self.yscale)
                if num_plot > 1 and legend_opt:
                    plt.legend()
                plot_main_name = f'{self.imdir}{data_dict["savename"]}'
                try:
                    plt.savefig(f'{plot_main_name}_{nuclide_i}_pcntdiff.png')
                except OverflowError:
                    print(f'Not plotting {plot_main_name} {nuclide_i} pcntdiff')
                plt.close()

        else:
            for i, x in enumerate(data_dict['xs']):
                if type(spatial_eval_node) != type(None):
                    try:
                        lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']
                        y = data_dict['ys'][i][:, spatial_eval_node, nuclide_i]
                    except IndexError:
                        continue
                    ending = f'spat{spatial_eval_node}'
                    plt.plot(x, y, label=lab)
                    num_plot += 1
            self._time_plot_helper(data_dict, nuclide_i, ending,
                                spatial_eval_node, num_plot)
        return




    def plot_time(self, data_dict, spatial_eval_node=None, print_diffs=True):
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
            if type(spatial_eval_node) == type(None):
                self._data_collector(data_dict, 'spat', None,
                                    'avg', nuclide_i, print_diffs,
                                    pcnt_diff_bool=True, print_data=False)
                self._data_collector(data_dict, 'in', None,
                                    'in', nuclide_i, print_diffs)
                self._data_collector(data_dict, 'ex', None,
                                    'ex', nuclide_i, print_diffs)
                self._data_collector(data_dict, 'gradient', None,
                                    'gradient', nuclide_i, print_diffs=False)
                self._data_collector(data_dict, 'std', None,
                                    'std', nuclide_i, print_diffs=False)
                self._data_collector(data_dict, 'max', None,
                                    'max', nuclide_i, print_diffs=False)

            if type(spatial_eval_node) != type(None):
                self._data_collector(data_dict, None, spatial_eval_node,
                                    None, nuclide_i, print_diffs)
        return
    
    def plot_space(self, data_dict, surf=True):
        num_nucs = []
        for i, x in enumerate(data_dict['xs']):
            num_nucs.append(np.shape(data_dict['ys'][i])[2])
        num_nucs = max(num_nucs)

        for nuclide_i in range(num_nucs):
            max_concs = list()
            min_concs = list()
            for i, x in enumerate(data_dict['xs']):
                cur_max_conc = np.max(data_dict['ys'][i][-1, :, nuclide_i])
                cur_min_conc = np.min(data_dict['ys'][i][-1, :, nuclide_i])
                max_concs.append(cur_max_conc)
                min_concs.append(cur_min_conc)
                try:
                    y = data_dict['ys'][i][-1, :, nuclide_i]
                except (KeyError, IndexError):
                    continue
                lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']

                plt.plot(self.run_params['positions'], y, label=lab,
                        marker='.')
            plt.legend()
            max_conc = np.max(max_concs)
            min_conc = np.min(min_concs)
            plt.xlabel('Space [cm]')
            plt.vlines(self.run_params['core_outlet'], 0, max_conc*1.1,
                        color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.ylim((min_conc*0.9, max_conc*1.1))
            plt.yscale(self.yscale) 
            plt.savefig(f'{self.imdir}final_time_{nuclide_i}.png')
            plt.close()

        if surf:
            i = data_dict['labs'].index('PDE')
            for nuclide_i in range(num_nucs):
                try:
                    x = data_dict['xs'][0]
                    y = self.run_params['positions']
                except (KeyError, IndexError):
                    continue
                data = dict()
                #x_name = r'Time $[s]$'
                x_name = data_dict['xlab']
                y_name = r'Length [cm]'
                z_name = r'Relative Concentration'
                
                
                data[x_name] = list()
                data[y_name] = list()
                data[z_name] = list()
                for xi, xval in enumerate(x):
                    for yi, yval in enumerate(y):
                        data[x_name].append(xval)
                        data[y_name].append(yval)
                        conc_at_point = data_dict['ys'][i][xi, yi, nuclide_i]
                        max_conc_at_time = np.max(data_dict['ys'][i][xi, :, nuclide_i])
                        data[z_name].append(conc_at_point / max_conc_at_time)


                df = pd.DataFrame.from_dict(data)
                pivotted = df.pivot(columns=x_name,index=y_name,values=z_name)
                color = sns.color_palette("magma", as_cmap=True)

                yticks = np.linspace(0, len(y)-1, 10, dtype=int)
                xticks = np.linspace(0, len(x)-1, 10, dtype=int)
                yticklabels = [f'{y[idx]:.1f}' for idx in yticks]
                xticklabels = [f'{x[idx]:.1f}' for idx in xticks]


                ax = sns.heatmap(pivotted, cmap=color,
                                xticklabels=xticklabels,
                                yticklabels=yticklabels,
                                norm='linear')
                                #norm=LogNorm(vmin=1e8, vmax=1e14))
                                #vmin=1e8,
                                #vmax=1e14)
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.set_xticklabels(xticklabels)
                ax.set_yticklabels(yticklabels)
                ax.invert_yaxis()
                plt.tight_layout()
                
                plt.savefig(f'{self.imdir}surf_{nuclide_i}.png')
                plt.close()
        return

    
    def gif_generate(self, data_dict):
        """
        Plots spatial conc over time where `data_dict` contains all
         necessary information

        Parameters
        ----------
        data_dict : dict
            key : str
                Variable name
        """
        start_time = time()
        from matplotlib.animation import FuncAnimation
        plt.rcParams['savefig.dpi'] = 100
        fig, ax = plt.subplots(1, 1)
        num_nucs = []
        for i, x in enumerate(data_dict['xs']):
            num_nucs.append(np.shape(data_dict['ys'][i])[2])
        num_nucs = max(num_nucs)
        max_conc = np.max(data_dict['ys'])

        def update(frame):
            ax.cla()
            plt.tight_layout()
            plt.xlabel('Space [cm]')
            plt.vlines(self.run_params['core_outlet'], 0, 1.05*max_conc,
                       color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.ylim((1e-1, 1.05*max_conc))
            plt.yscale(self.yscale)

            for nuclide_i in range(num_nucs):
                for i, x in enumerate(data_dict['xs']):
                    try:
                        y = data_dict['ys'][i][frame, :, nuclide_i]
                    except KeyError:
                        continue
                    lab = data_dict[f'lab_method{i}_nuc{nuclide_i}']

                    ax.plot(self.run_params['positions'], y, label=lab,
                            marker='.')
                    ax.set_title(f'Time: {round(frame*self.run_params["dt"], 4)} s')
                    plt.legend()
            
        net_animation_time = 10*1000
        frame_count = len(self.run_params['times'])
        animation = FuncAnimation(fig, update,
                                  frames=frame_count,
                                  interval=net_animation_time/frame_count,
                                  repeat_delay=3000)
        animation.save(f'{self.imdir}isobar_evolution.gif', writer='pillow')
        plt.close()
        plt.rcParams['savefig.dpi'] = 300
        print(f'Gif creation took: {round(time() - start_time, 3)} s')
        return
    
    def write_csv(self, data_dict):
        num_nucs = []
        for i, x in enumerate(data_dict['xs']):
            num_nucs.append(np.shape(data_dict['ys'][i])[2])
        num_nucs = max(num_nucs)
        for nuclide_i in range(num_nucs):
            for i, x in enumerate(data_dict['xs']):
                df = pd.DataFrame(data_dict['ys'][i][:, :, nuclide_i])
                df.insert(loc=0, column='Time [s]', value=self.run_params['times'])
                df.to_csv(f'{self.imdir}data_{nuclide_i}.csv', index=False)
        return
    
    def plot_gen(self, data_dict, spatial_eval_positions=[],
                 time_eval_positions=[],
                 surf_opt=True):
        """
        Generates various plots based on plotting parameters

        Parameters
        ----------
        data_dict : dict
            key : str
                Name of variable
        spatial_eval_positions : list of int (optional)
            Spatial index positions to evaluate over time
        
        """
        if not self.plotting_params['plotting']:
            return
        self.plot_obj = PlotHolder()
        self.plot_dict = {}
        self.plot_dict['data'] = {}
        self.plot_time(data_dict)
        if self.plotting_params['msre']:
            self.plot_obj.plot_data(self.imdir)
        #self.write_csv(data_dict)
        self.plot_space(data_dict, surf=surf_opt)
        for pos in spatial_eval_positions:
            self.plot_time(data_dict, pos) 
        if self.plotting_params['gif']:
            self.gif_generate(data_dict)
        if self.plotting_params['parasitic_absorption']:
            self.parasitic_plot(data_dict)
        
        return


