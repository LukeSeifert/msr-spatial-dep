import solvers
import os

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
        Determines time scaling to use

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
            if tf < time:
                continue
            xlab = f'Time [{label_list[ti]}]'
            scaling_factor = 1/time
            return scaling_factor, xlab
        
    def _method_change(self, methods, method_name, x_vals, xfactor=1):
        """
        Used to handle concentration plotting comparisons with a single
            method change.

        Parameters
        ----------
        methods : list
            List of methods to apply
        method_name : str
            Key for dictionary to apply method to `run_params`
        x_vals : :class:`np.ndarray`
            x-axis values
        xfactor : float
            Value to scale `x_vals` by

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
        for method in methods:
            self.run_params[method_name] = method
            xs.append(xfactor * x_vals)
            result_matrix = solvers.DiffEqSolvers(self.run_params, self.data_params).result_mat
            ys.append(result_matrix)
            labs.append(method)
        return xs, ys, labs

    def _save_data(self, xs, ys, labs, xlab, ylab, savename):
        self.data = {}
        self.data['xs'] = xs
        self.data['ys'] = ys
        self.data['labs'] = labs
        self.data['xlab'] = xlab
        self.data['ylab'] = ylab
        self.data['savename'] = savename
        return
    
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
        methods = ['PDE', 'ODE']
        ylab = 'Concentration [atoms/cc]'
        savename = 'PDE_ODE_comparison'
        time_factor, xlab = self._time_lab()
        xs, ys, labs = self._method_change(methods, 'solver_method',
                                           self.run_params['times'],
                                           time_factor)
        self._save_data(xs, ys, labs, xlab, ylab, savename)

        self.run_params['solver_methods'] = current_solver
        data = self.data
        return data