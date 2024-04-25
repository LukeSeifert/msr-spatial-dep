import numpy as np

class DiffEqSolvers:
    def __init__(self, run_params, data_params, run=True):
        """
        This class allows for the solve of a system of PDEs by
        solving each individually in a Jacobi-like manner.
        This approach will provide a more accurate result than an ODE appraoch,
        as contributions from other nuclides in the isobar will be included.
        The next step up from this approach is to incorporate
        the spatial solve within the depletion solver itself.

        Parameters
        ----------
        run_params : dict
            key : str
                Name of run parameter
        data_params : dict
            ket : str
                Name of data parameter

        """
        self.spacenodes = run_params['spacenodes']
        self.num_nucs = run_params['num_nuclides']
        self.final_time = run_params['final_time']
        run_params['frac_out'] = 1 - run_params['frac_in']
        run_params['core_outlet']   = (run_params['net_length'] * run_params['frac_out'])
        run_params['excore_outlet'] = run_params['core_outlet'] + (run_params['net_length'] * run_params['frac_in'])
        self.z_excore_outlet = run_params['excore_outlet']
        self.z_core_outlet = run_params['core_outlet']
        self.CFL_cond = run_params['CFL_cond']
        
        run_params['incore_volume'] = run_params['net_cc_vol'] * run_params['frac_in']
        linear_flow_rate = (run_params['vol_flow_rate'] / (run_params['fuel_fraction'] * np.pi * (run_params['core_rad'])**2))
        run_params['incore_flowrate'] = linear_flow_rate
        run_params['excore_flowrate'] = linear_flow_rate
        run_params['max_flowrate'] = max(run_params['incore_flowrate'], run_params['excore_flowrate'])
        run_params['dz'] = np.diff(np.linspace(0, run_params['excore_outlet'], run_params['spacenodes']))[0]
        run_params['positions'] = np.linspace(0, run_params['excore_outlet'], run_params['spacenodes'])
        run_params['dt'] = run_params['CFL_cond'] * run_params['dz'] / run_params['max_flowrate']
        run_params['times'] = np.arange(0, run_params['final_time']+run_params['dt'], run_params['dt'])

        self.incore_flowrate = run_params['incore_flowrate']
        self.excore_flowrate = run_params['excore_flowrate']
        self.incore_volume = run_params['incore_volume']
        self.dz = run_params['dz']
        self.positions = run_params['positions']
        self.dt = run_params['dt']
        self.times = run_params['times']

        self.flow_vec = self._format_spatial(self.incore_flowrate,
                                           self.excore_flowrate)

        self.lams = data_params['lams']
        self.loss_rates = data_params['loss_rates']
        self.dec_fracs = data_params['dec_frac']
        self.FYs = data_params['FYs']

        self.mu = {}
        for nuclide in range(self.num_nucs):
            incore_losses = self.lams[nuclide] + self.loss_rates[nuclide]
            excore_losses = self.lams[nuclide]
            cur_nuc_losses = self._format_spatial(incore_losses, excore_losses)
            self.mu[nuclide] = cur_nuc_losses

        self.S = {}
        self.run_params = run_params

        if run:
            if run_params['solver_method'] == 'PDE':
                self.res_mat = self.PDE_solver()
            elif run_params['solver_method'] == 'ODE':
                self.res_mat = self.ode_solve()
        
        return
    
    def _format_spatial(self, term1, term2):
        """
        Distribute term 1 in < z_core_outlet and term 2 above outlet.
        Returns a list of terms corresponding to each z.

        Parameters
        ----------
        term1 : float or :class:`np.ndarray`
            Term in the in-core region
        term2 : float or :class:`np.ndarray`
            Term in the ex-core region

        Returns
        -------
        return_list : :class:`np.ndarray`
            Spatial distribution of values at each point

        """
        return_list = np.zeros(self.spacenodes)
        if np.size(term1) > 1:
            vector_form = True
        else:
            vector_form = False

        for zi, z in enumerate(self.positions):
            if vector_form:
                incore_term = term1[zi]
                excore_term = term2[zi]
            else:
                incore_term = term1
                excore_term = term2

            if z <= self.z_core_outlet:
                return_list[zi] = incore_term
            elif z > self.z_core_outlet:
                return_list[zi] = excore_term

        return np.asarray(return_list)

    def _initialize_concs(self):
        """
        Set up the 1D concentrations

        """
        self.concs = []
        for nuclide in range(self.num_nucs):
            self.concs.append(np.zeros(self.spacenodes))
        return
    
    def _initialize_result_mat(self):
        """
        Set up the 3D result matrix with the form
            time, space, nuclide

        Returns
        -------
        result_mat : :class:`np.ndarray`
            Holds values over time, space, and nuclide (in that order)

        """
        result_mat = np.zeros((len(self.times), self.spacenodes, self.num_nucs))
        for nuclide in range(self.num_nucs):
            result_mat[0, :, nuclide] = self.concs[nuclide]
        return result_mat

    def _update_sources(self):
        """
        Update source terms based on concentrations

        """
        for gain_nuc in range(self.num_nucs):
            fission_source = self.FYs[gain_nuc]/self.incore_volume
            decay_source = np.zeros(len(self.concs[gain_nuc]))
            for loss_nuc in range(self.num_nucs):
                try:
                    frac = self.dec_fracs[(loss_nuc, gain_nuc)]
                    decay_source += (frac * self.concs[loss_nuc] * self.lams[loss_nuc])
                except KeyError:
                    continue
            incore_source = fission_source + decay_source
            excore_source = decay_source
            cur_source = self._format_spatial(incore_source, excore_source)
            self.S[gain_nuc] = cur_source
        return

    def _update_result_mat(self, result_mat, time_index):
        """
        Updates the result matrix with new concentrations

        Parameters
        ----------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        time_index : int
            Current time index

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        """
        for nuclide in range(self.num_nucs):
            result_mat[time_index, :, nuclide] = self.concs[nuclide]
        return result_mat

    def _external_ODE_no_step(self, conc, nuclide_index):
        """
        This function applies a single time step iteration of the ODE

        Parameters
        ----------
        conc : float
            Initial concentration
        nuclide_index : int
            Nuclide index

        Returns
        -------
        conc : float
            Concentration at current time
        """
        conc = (conc + self.dt * (self.S[nuclide_index][0] -
                                  self.mu[nuclide_index][0] * conc))

        return conc

    def _external_PDE_no_step(self, conc, nuclide_index):
        """
        This function applies a single time step iteration of the PDE

        Parameters
        ----------
        conc : :class:`np.ndarray`
            Concentration over spatial nodes at previous time
        isotope : int
            Nuclide isobar indicator

        Returns
        -------
        conc : 1D vector
            Concentration over spatial nodes at current time
        """
        S_vec = self.S[nuclide_index]
        mu_vec = self.mu[nuclide_index]
        J = np.arange(0, self.spacenodes)
        Jm1 = np.roll(J,  1)
        dz = np.diff(self.positions)[0]

        conc_mult = 1 - mu_vec * self.dt
        add_source = S_vec * self.dt
        lmbda = (self.flow_vec * self.dt / dz)
        conc = add_source + conc_mult * conc + self.CFL_cond * (conc[Jm1] - conc)

        return conc

    def ode_solve(self):
        """
        Solve the time dependent ODE

        """
        self._initialize_concs()
        ODE_result_mat = self._initialize_result_mat()
        for ti, t in enumerate(self.times[1:]):
            self._update_sources()

            for nuclide in range(self.num_nucs):
                self.concs[nuclide] = self._external_ODE_no_step(self.concs[nuclide], nuclide)

            ODE_result_mat = self._update_result_mat(ODE_result_mat, ti+1)
        self.result_mat = ODE_result_mat
        return ODE_result_mat
    
    def PDE_solver(self):
        """
        Runs the PDE solver to generate the time and space dependent
            concentrations for each nuclide.
        
        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        
        """
        self._initialize_concs()
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(self.times[:-1]):

            self._update_sources()

            for nuclide in range(self.num_nucs):
                self.concs[nuclide] = self._external_PDE_no_step(self.concs[nuclide], nuclide)
            result_mat = self._update_result_mat(result_mat, ti+1)
        self.result_mat = result_mat
        return result_mat
    