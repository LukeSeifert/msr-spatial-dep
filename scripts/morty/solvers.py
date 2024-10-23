import numpy as np
from time import time
import matplotlib.pyplot as plt

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
        run : bool (optional)
            Run a solver immediately

        """
        self.spacenodes = run_params['spacenodes']
        self.num_nucs = run_params['num_nuclides']
        self.final_time = run_params['final_time']
        run_params['frac_out'] = 1 - run_params['frac_in']
        run_params['core_outlet'] = (
            run_params['net_length'] *
            run_params['frac_in'])
        run_params['excore_outlet'] = run_params['net_length']
        self.z_excore_outlet = run_params['excore_outlet']
        self.z_core_outlet = run_params['core_outlet']

        run_params['incore_volume'] = run_params['net_cc_vol'] * \
            run_params['frac_in']
        #linear_flow_rate = (run_params['vol_flow_rate'] /
        #                    (run_params['fuel_fraction'] *
        #                     np.pi *
        #                     (run_params['core_rad'])**2))
        run_params['incore_flowrate'] = run_params['linear_flow_rate']
        run_params['excore_flowrate'] = run_params['linear_flow_rate']
        run_params['max_flowrate'] = max(
            run_params['incore_flowrate'],
            run_params['excore_flowrate'])
        run_params['dz'] = np.diff(
            np.linspace(
                0,
                run_params['excore_outlet'],
                run_params['spacenodes']))[0]
        run_params['positions'] = np.linspace(
            0, run_params['excore_outlet'], run_params['spacenodes'])
        
        #run_params['dt'] = run_params['final_time'] / run_params['num_times']
        #run_params['CFL_cond'] = (run_params['dt'] * run_params['max_flowrate'] / run_params['dz'])
        run_params['dt'] = run_params['dz'] * run_params['CFL_cond'] / run_params['max_flowrate']
        self.CFL_cond = run_params['CFL_cond']
        if self.CFL_cond > 0.9:
            print(f'{run_params["CFL_cond"] = }')
        run_params['times'] = np.arange(
            0,
            run_params['final_time'] +
            run_params['dt'],
            run_params['dt'])
        run_params['power_W'] = self._power_hist(version=run_params['power_version'],
                                        times=run_params['times'],
                                        p0=run_params['p0'])



        self.incore_flowrate = run_params['incore_flowrate']
        self.excore_flowrate = run_params['excore_flowrate']
        self.incore_volume = run_params['incore_volume']
        self.dz = run_params['dz']
        self.positions = run_params['positions']
        self.dt = run_params['dt']
        self.times = run_params['times']
        self.power = run_params['power_W']
        self.p0 = run_params['p0']

        self.flow_vec = self._format_spatial(self.incore_flowrate,
                                             self.excore_flowrate)

        self.lams = data_params['lams']
        self.loss_rates = data_params['loss_rates']
        self.dec_fracs = data_params['dec_frac']
        self.FYs = data_params['FYs']
        self.reprs = data_params['repr_rates']

        self.mu = {}
        for nuclide in range(self.num_nucs):
            incore_losses = self.lams[nuclide] + self.loss_rates[nuclide]
            excore_losses = self.lams[nuclide] + self.reprs[nuclide]
            cur_nuc_losses = self._format_spatial(incore_losses, excore_losses)
            self.mu[nuclide] = cur_nuc_losses

        self.S = {}
        self.run_params = run_params

        if run:
            start = time()
            if run_params['solver_method'] == 'PDE':
                self.res_mat = self.pde_solve()
            elif run_params['solver_method'] == 'ODE':
                self.res_mat = self.ode_solve()
            took = time() - start
            print(f'Took {round(took, 3)} seconds')

        return
    
    def _power_hist(self, version, times, p0):
        power_vals = list()

        def _time_sorter(coarse_x, coarse_y, fine_x, scale=p0):
            fine_y = []
            for x in fine_x:
                y = None
                for i in range(len(coarse_x)-1):
                    if coarse_x[i] <= x < coarse_x[i+1]:
                        y = scale * coarse_y[i]
                        break
                fine_y.append(y)
            return fine_y


        if version == 'constant':
            for t in times:
                power_vals.append(p0)
        elif version == 'sin':
            for t in times:
                power = p0/2 * (np.sin(np.pi * t / 30) + 1)
                power_vals.append(power)
        elif version == 'neg_exp':
            for t in times:
                power = p0 * np.exp(-t)
                power_vals.append(power)
        elif version == 'step':
            pulse_times = [
        0, 1.25*24*3600, 1.25*24*3600+10*60, 1.25*24*3600*1e6
]
            pulse_rel_powers = [
          1,           10,               10
]
            power_vals = _time_sorter(pulse_times, pulse_rel_powers, times)

        elif version == 'msre':
            newts = [0.0, 0.4166667, 1.0, 1.5, 2.0, 3.166667, 77.0, 80.0, 81.0, 82.25, 84.0, 86.0, 87.0, 87.5, 88.0, 88.5, 91.0, 95.0, 105.0, 108.0, 108.5, 112.0, 113.75, 115.0, 116.75, 122.5, 140.0, 141.0, 141.5, 153.5, 157.75, 163.5, 170.5, 171.5, 174.0, 262.3333, 265.0, 265.25, 280.0, 287.8333, 292.8333, 293.8333, 295.0, 296.25, 297.25, 324.9167, 338.5]
            msre_times = np.asarray(newts) * 24 * 3600
            msre_rel_powers = [0.03406, 0.0, 0.0681199, 0.0, 0.1362398, 0.0, 0.1362398, 0.0, 0.3405995, 0.0, 0.3405995, 0.0, 0.681199, 0.0, 0.681199, 0.0, 0.681199, 0.0, 0.722071, 0.0, 0.8855586, 0.0, 0.681199, 0.0, 0.9019073, 0.0, 1.0, 0.0, 0.9888076, 0.0, 1.0, 1.0, 0.0, 0.762943, 0.0, 0.7259027, 0.0, 0.8174387, 0.0, 0.9224797, 0.0, 1.0, 0.0, 1.0, 0.0, 0.8855586, 1.0]
            power_vals = _time_sorter(msre_times, msre_rel_powers, times)

        plt.step(times/(24*3600), power_vals, where='post')
        plt.xlabel('Time [d]')
        plt.ylabel('Power [W]')
        plt.savefig('images/power_hist.png')
        plt.close()

        return power_vals



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
        result_mat = np.zeros(
            (len(self.times), self.spacenodes, self.num_nucs))
        for nuclide in range(self.num_nucs):
            result_mat[0, :, nuclide] = self.concs[nuclide]
        return result_mat

    def _update_sources(self, ti):
        """
        Update source terms based on concentrations

        Parameters
        ----------
        ti : int
            Current time index

        """
        for gain_nuc in range(self.num_nucs):
            fission_source = self.power[ti]/self.p0 * self.FYs[gain_nuc]  # fiss/cc-s
            decay_source = np.zeros(len(self.concs[gain_nuc]))
            for loss_nuc in range(self.num_nucs):
                try:
                    frac = self.dec_fracs[(loss_nuc, gain_nuc)]
                    decay_source += (frac *
                                     self.concs[loss_nuc] *
                                     self.lams[loss_nuc])
                except KeyError:
                    continue
            incore_source = fission_source + decay_source
            excore_source = decay_source
            cur_source = self._format_spatial(incore_source, excore_source)
            self.S[gain_nuc] = cur_source
        return
    
    def _update_losses(self, ti):
        """
        Update loss terms based on power history

        Parameters
        ----------
        ti : int
            Current time index

        """
        for nuclide in range(self.num_nucs):
            incore_losses = (self.lams[nuclide] + 
                             self.power[ti]/self.p0 * self.loss_rates[nuclide])
            excore_losses = self.lams[nuclide] + self.reprs[nuclide]
            cur_nuc_losses = self._format_spatial(incore_losses, excore_losses)
            self.mu[nuclide] = cur_nuc_losses
        return

    def _update_result_mat(self, result_mat, time_index):
        """
        Updates the result matrix with new concentrations

        Parameters
        ----------
        result_mat : :class:`np.ndarray`
            Holds values over time, space, and nuclide (in that order)
        time_index : int
            Current time index

        Returns
        -------
        result_mat : :class:`np.ndarray`
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
        #conc = (conc + self.dt * (self.S[nuclide_index][0] -
        #                          self.mu[nuclide_index][0] * conc))
        #print(self.S[nuclide_index][0])
        #print(self.mu[nuclide_index][0])
        #print(self.mu[nuclide_index][-1])
        #input()
        conc = ((conc + self.S[nuclide_index][0] * self.dt) / (1 + self.mu[nuclide_index][0] * self.dt))


        return conc

    def _external_PDE_no_step(self, conc, nuclide_index):
        """
        This function applies a single time step iteration of the PDE

        Parameters
        ----------
        conc : :class:`np.ndarray`
            Concentration over spatial nodes at previous time
        nuclide_index : int
            Nuclide isobar indicator

        Returns
        -------
        conc : :class:`np.ndarray`
            Concentration over spatial nodes at current time
        """
        S_vec = self.S[nuclide_index]
        mu_vec = self.mu[nuclide_index]
        J = np.arange(0, self.spacenodes)
        Jm1 = np.roll(J, 1)
        dz = np.diff(self.positions)[0]

        conc_mult = 1 - mu_vec * self.dt
        add_source = S_vec * self.dt
        CFL_vec = (self.flow_vec * self.dt / dz)
        #conc = add_source + conc_mult * conc + CFL_vec * (conc[Jm1] - conc)
        #print(S_vec[0])
        #print(S_vec[-1])
        #print(mu_vec[0])
        #print(mu_vec[-1])
        #input()
        #conc = conc + self.dt * (S_vec - mu_vec*conc - self.flow_vec * (conc - conc[Jm1])/dz)
        conc = ((conc + self.dt * (S_vec + self.flow_vec/dz * (conc[Jm1] - conc))) / (1 + mu_vec * self.dt))


        return conc

    def ode_solve(self):
        """
        Solve the time dependent ODE

        Returns
        -------
        ODE_result_mat : :class:`np.ndarray`
            Holds concentrations over time, space, and nuclide (in that order)

        """
        self._initialize_concs()
        ODE_result_mat = self._initialize_result_mat()
        for ti, t in enumerate(self.times[1:]):
            self._update_sources(ti)
            self._update_losses(ti)

            for nuclide in range(self.num_nucs):
                self.concs[nuclide] = self._external_ODE_no_step(
                    self.concs[nuclide], nuclide)

            ODE_result_mat = self._update_result_mat(ODE_result_mat, ti + 1)
        self.result_mat = ODE_result_mat
        return ODE_result_mat

    def pde_solve(self):
        """
        Runs the PDE solver to generate the time and space dependent
            concentrations for each nuclide.

        Returns
        -------
        result_mat : :class:`np.ndarray`
            Holds values over time, space, and nuclide (in that order)

        """
        self._initialize_concs()
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(self.times[:-1]):
            self._update_sources(ti)
            self._update_losses(ti)

            for nuclide in range(self.num_nucs):
                self.concs[nuclide] = self._external_PDE_no_step(
                    self.concs[nuclide], nuclide)
            result_mat = self._update_result_mat(result_mat, ti + 1)
        self.result_mat = result_mat
        return result_mat
