import solvers
import numpy as np
from data import DataHandler
from scipy import integrate


class AnalysisCollection:
    def __init__(self, analysis_params, run_params, data_params):
        """
        This class contains a collection of analysis procedures which can be run.
        The data is returned in an easily-plottable dictionary as well as saved to
        text files for future analysis.

        Parameters
        ----------
        analysis_params : dict
            key : str
                Name of analysis parameter
        run_params : dict
            key : str
                Name of run parameter
        data_params : dict
            ket : str
                Name of data parameter
        """
        self.analysis_params = analysis_params
        self.run_params = run_params
        self.data_params = data_params
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
        time_list = [year, day, hour, minute, second, millisecond]
        label_list = ['yr', 'd', 'hr', 'min', 's', 'ms']
        for ti, time in enumerate(time_list):
            if time > tf:
                continue
            xlab = f'Time [{label_list[ti]}]'
            scaling_factor = 1 / time
            return scaling_factor, xlab

    def _method_change(
            self,
            methods,
            method_name,
            xfactor=1,
            methodname_extension=''):
        """
        Used to handle concentration plotting comparisons with a single
            method change.

        Parameters
        ----------
        methods : list
            List of methods to apply
        method_name : str
            Key for dictionary to apply method to `run_params`
        xfactor : float
            Value to scale `x_vals` by
        methodname_extension : str (optional)
            String to add to method name in plotting

        Returns
        -------
        xs : list of list
            Contains the `x` values to be plotted
        ys : list of list
            Contains the `y` values to be plotted
        labs : list of list
            Contains the ordered labels to use in plotting


        """
        xs, ys, labs = [], [], []
        self.parasitic_results = []
        for method in methods:
            self.run_params[method_name] = method
            self.data_params = DataHandler(self.run_params).data_params
            solver = solvers.DiffEqSolvers(
                self.run_params, self.data_params, run=False)
            x_vals = solver.run_params['times']
            xs.append(xfactor * x_vals)
            result_matrix = solvers.DiffEqSolvers(
                self.run_params, self.data_params).result_mat
            ys.append(result_matrix)
            labs.append(f'{method}{methodname_extension}')
            self.parasitic_results.append(self._parasitic_abs(result_matrix))
        return xs, ys, labs

    def _save_data(self, xs, ys, labs, xlab, ylab, savename):
        """
        Saves the collected data into a single dictionary

        Parameters
        ----------
        xs : list of list
            Contains the `x` values to be plotted
        ys : list of list
            Contains the `y` values to be plotted
        labs : list of list
            Contains the ordered labels to use in plotting
        xlab : str
            Name of xlabel
        ylab : str
            Name of ylabel
        savename : str
            Name of file to save as


        """
        self.data = {}
        self.data['xs'] = xs
        self.data['ys'] = ys
        self.data['labs'] = labs
        self.data['xlab'] = xlab
        self.data['ylab'] = ylab
        self.data['savename'] = savename
        self.data['parasitic'] = self.parasitic_results
        num_nucs = []
        for i, x in enumerate(self.data['xs']):
            num_nucs.append(np.shape(self.data['ys'][i])[2])

        num_nucs = max(num_nucs)
        for nuclide_i in range(num_nucs):
            final_data_points = []
            final_names = []
            for i, x in enumerate(self.data['xs']):
                nuclide = self.data_params['tracked_nucs'][nuclide_i]
                try:
                    y = np.mean(self.data['ys'][i][:, :, nuclide_i], axis=1)
                except IndexError:
                    continue
                lab = f'{self.data["labs"][i]} {nuclide}'
                self.data[f'spat_avg_y_method{i}_nuc{nuclide_i}'] = y
                self.data[f'lab_method{i}_nuc{nuclide_i}'] = lab
                final_data_points.append(y[-1])
                final_names.append(lab)

            for data_i, data in enumerate(final_data_points):
                pcnt_diff = (
                    data - final_data_points[-1]) / (final_data_points[-1]) * 100
                print(
                    f'Percent Diff of {final_names[data_i]} from most accurate: {pcnt_diff}%')
        return

    def _parasitic_abs(self, result_mat):
        """
        Calculates the amount of parasitic absorption for each nuclide
            as atoms per cc as a function of time

        Parameters
        ----------
        result_mat : :class:`np.ndarray`
            Holds values over time, space, and nuclide (in that order)

        Returns
        -------
        parasitic_abs_val : :class:`np.ndarray`
            Parasitic absorption rates for nuclides over time in
            atoms per cc
        """
        parasitic_abs_val = []
        for nuclide in range(self.run_params['num_nuclides']):
            paras_rate = []
            for ti, t in enumerate(self.run_params['times']):
                spat_avg_conc = np.mean(result_mat[ti, :, nuclide])
                parasitic = spat_avg_conc * \
                    self.data_params['loss_rates'][nuclide]
                paras_rate.append(parasitic)
            integral_form = integrate.cumtrapz(
                paras_rate, self.run_params['times'])
            integral_form = np.insert(integral_form, 0, 0.0)
            parasitic_abs_val.append(integral_form)
        return parasitic_abs_val

    def ode_pde_compare(self):
        """
        Run with current parameters for both ODE and PDE solver methods.

        Returns
        -------
        data : dict
            key : str
                Name of variable
        """
        current_solver = self.run_params['solver_method']
        methods = ['ODE', 'PDE']
        ylab = 'Concentration [atoms/cc]'
        savename = 'PDE_ODE_comparison'
        time_factor, xlab = self._time_lab()
        xs, ys, labs = self._method_change(methods, 'solver_method',
                                           time_factor)
        self._save_data(xs, ys, labs, xlab, ylab, savename)

        self.run_params['solver_methods'] = current_solver
        data = self.data
        return data

    def nuclide_refinement(self, max_nuc=5):
        """
        Run with current parameters for varying number of nuclides.

        Parameters
        ----------
        max_nuc : int
            Maximum number of nuclides

        Returns
        -------
        data : dict
            key : str
                Name of variable
        """
        current_nuclides = self.run_params['num_nuclides']
        methods = np.linspace(1, max_nuc, max_nuc).astype(int)
        ylab = 'Concentration [atoms/cc]'
        savename = 'nuclide_refinement'
        time_factor, xlab = self._time_lab()
        xs, ys, labs = self._method_change(methods, 'num_nuclides',
                                           time_factor,
                                           methodname_extension=' nuclides')
        self._save_data(xs, ys, labs, xlab, ylab, savename)

        self.run_params['num_nuclides'] = current_nuclides
        data = self.data
        return data

    def spatial_refinement(self, spat_node_list=[10, 100, 1000, 10000]):
        """
        Run with current parameters for varying number of spatial nodes.

        Parameters
        ----------
        spat_node_list : list of int
            Number of spatial nodes for each run

        Returns
        -------
        data : dict
            key : str
                Name of variable
        """
        current_spacenodes = self.run_params['spacenodes']
        methods = spat_node_list
        ylab = 'Concentration [atoms/cc]'
        savename = 'spatial_refinement'
        time_factor, xlab = self._time_lab()
        xs, ys, labs = self._method_change(methods, 'spacenodes',
                                           time_factor,
                                           methodname_extension=' nodes')
        self._save_data(xs, ys, labs, xlab, ylab, savename)

        self.run_params['spacenodes'] = current_spacenodes
        data = self.data
        return data
