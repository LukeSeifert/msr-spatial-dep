import numpy as np
from time import time
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings

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
        print(f'Spatial discretization: {run_params["dz"]} cm')
        print(f'Time step: {run_params["dt"]} s')
        #run_params['dt'] = (run_params['dz'] / run_params['max_flowrate'])
        print(f'Number of time steps: {int(run_params["final_time"] / run_params["dt"])}')
        self.CFL_cond = run_params['CFL_cond']
        #if self.CFL_cond > 0.9:
        #    print(f'{run_params["CFL_cond"] = }')
        run_params['times'] = np.arange(
            0,
            run_params['final_time'] +
            run_params['dt'],
            run_params['dt'])
        run_params['reduced_times'] = run_params['times'][::run_params['time_mult']]
        run_params['reduced_times'] = np.append(
            run_params['reduced_times'],
            run_params['final_time'])
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
        self.reduced_times = run_params['reduced_times']
        self.power = run_params['power_W']
        self.p0 = run_params['p0']
        self.run_params = run_params
        self.repr_loc = run_params['repr_loc']

        for xi, x in enumerate(self.positions):
            if x > self.z_core_outlet:
                self.transition_index = xi
                break

        self.flow_vec = self._format_spatial(self.incore_flowrate,
                                             self.excore_flowrate)
        self.base_flow = deepcopy(self.flow_vec)

        self.lams = data_params['lams']
        self.loss_rates = data_params['loss_rates']
        self.dec_fracs = data_params['dec_frac']
        self.FYs = data_params['FYs']
        self.reprs = data_params['repr_rates']

        self._avg_plot_flow()

        self.mu = {}
        for nuclide in range(self.num_nucs):

            incore_losses = self.lams[nuclide] + self.loss_rates[nuclide]
            excore_losses = self.lams[nuclide]
            if self.repr_loc == 'in':
                incore_losses += self.reprs[nuclide]
            elif self.repr_loc == 'ex':
                excore_losses += self.reprs[nuclide]
            cur_nuc_losses = self._format_spatial(incore_losses, excore_losses)
            self.mu[nuclide] = cur_nuc_losses

        self.S = {}
        self.run_params = run_params
        self.fission_shape = self._incore_spatial_source()

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
                power = p0/2 * (np.sin(np.pi * t / (10/2) + 2*np.pi) + 1)
                power_vals.append(power)
        elif version == 'sqwv':
            pulse_length = 10
            for t in times:
                if t % pulse_length < pulse_length/2:
                    power = p0
                else:
                    power = 0
                power_vals.append(power)
        elif version == 'neg_exp':
            for t in times:
                power = p0 * np.exp(-t)
                power_vals.append(power)
        elif version == 'step':
            pulse_times = [
        #0, 1.25*24*3600, 1.25*24*3600+10*60, 1.25*24*3600*1e6
        0, 3600, 3600+10*60, 3600*1e6
]
            pulse_rel_powers = [
          1,           1e1,               1e1
]
            power_vals = _time_sorter(pulse_times, pulse_rel_powers, times)

        elif version == 'msre':
            msre_times = [
            0.00000E+00,
            3.60000E+04,
            3.60000E+04,
            8.64000E+04,
            8.64000E+04,
            1.29600E+05,
            1.29600E+05,
            1.72800E+05,
            1.72800E+05,
            2.73600E+05,
            2.73600E+05,
            6.65280E+06,
            6.65280E+06,
            6.91200E+06,
            6.91200E+06,
            6.99840E+06,
            6.99840E+06,
            7.10640E+06,
            7.10640E+06,
            7.25760E+06,
            7.25760E+06,
            7.43040E+06,
            7.43040E+06,
            7.51680E+06,
            7.51680E+06,
            7.56000E+06,
            7.56000E+06,
            7.60320E+06,
            7.60320E+06,
            7.64640E+06,
            7.64640E+06,
            7.86240E+06,
            7.86240E+06,
            8.20800E+06,
            8.20800E+06,
            9.07200E+06,
            9.07200E+06,
            9.33120E+06,
            9.33120E+06,
            9.37440E+06,
            9.37440E+06,
            9.67680E+06,
            9.67680E+06,
            9.82800E+06,
            9.82800E+06,
            9.93600E+06,
            9.93600E+06,
            1.00872E+07,
            1.00872E+07,
            1.05840E+07,
            1.05840E+07,
            1.20960E+07,
            1.20960E+07,
            1.21824E+07,
            1.21824E+07,
            1.22256E+07,
            1.22256E+07,
            1.32624E+07,
            1.32624E+07,
            1.36296E+07,
            1.36296E+07,
            1.41264E+07,
            1.41264E+07,
            1.47312E+07,
            1.47312E+07,
            1.48176E+07,
            1.48176E+07,
            1.50336E+07,
            1.50336E+07,
            2.26656E+07,
            2.26656E+07,
            2.28960E+07,
            2.28960E+07,
            2.29176E+07,
            2.29176E+07,
            2.41920E+07,
            2.41920E+07,
            2.48688E+07,
            2.48688E+07,
            2.53008E+07,
            2.53008E+07,
            2.53872E+07,
            2.53872E+07,
            2.54880E+07,
            2.54880E+07,
            2.55960E+07,
            2.55960E+07,
            2.56824E+07,
            2.56824E+07,
            2.80728E+07,
            2.80728E+07,
            2.92464E+07,
            2.92464E+07,
            3.02832E+07,
            3.02832E+07,
            3.06720E+07,
            3.06720E+07,
            3.19680E+07,
            3.19680E+07,
            3.20688E+07,
            3.20688E+07,
            3.21408E+07,
            3.21408E+07,
            3.33072E+07,
            3.33072E+07,
            3.39984E+07,
            3.39984E+07,
            3.46572E+07,
            3.46572E+07,
            3.46896E+07,
            3.46896E+07,
            3.51936E+07,
            3.51936E+07,
            3.53808E+07,
            3.53808E+07,
            3.54672E+07,
            3.54672E+07,
            3.87504E+07,
            3.87504E+07,
            3.97008E+07,
            3.97008E+07,
            3.99600E+07,
            3.99600E+07,
            4.03056E+07,
            4.03056E+07,
            4.05648E+07,
            4.05648E+07,
            4.07376E+07,
            4.07376E+07,
            4.43664E+07,
            4.43664E+07,
            4.44960E+07,
            4.44960E+07,
            4.45392E+07,
            4.45392E+07,
            4.46688E+07,
            4.46688E+07,
            4.48416E+07,
            4.48416E+07,
            4.61376E+07,
            4.61376E+07,
            4.61808E+07,
            4.61808E+07,
            4.66128E+07,
            4.66128E+07,
            4.73040E+07,
            4.73040E+07,
            4.84992E+07,
            4.84992E+07,
            5.18184E+07,
            5.18184E+07,
            5.20344E+07,
            5.20344E+07,
            5.23152E+07,
            5.23152E+07,
            5.50944E+07,
            5.50944E+07,
            5.53392E+07,
            5.53392E+07,
            5.61168E+07,
            5.61168E+07,
            5.63760E+07,
            5.63760E+07,
            5.71104E+07,
            5.71104E+07,
            5.78664E+07,
            5.78664E+07,
            5.87952E+07,
            5.87952E+07,
            5.94864E+07,
            5.94864E+07,
            6.29424E+07
            ]
            msre_rel_powers = [
            3.40600E-02,
            3.40600E-02,
            0.00000E+00,
            0.00000E+00,
            6.81199E-02,
            6.81199E-02,
            0.00000E+00,
            0.00000E+00,
            1.36240E-01,
            1.36240E-01,
            0.00000E+00,
            0.00000E+00,
            1.36240E-01,
            1.36240E-01,
            0.00000E+00,
            0.00000E+00,
            3.40600E-01,
            3.40600E-01,
            0.00000E+00,
            0.00000E+00,
            3.40600E-01,
            3.40600E-01,
            0.00000E+00,
            0.00000E+00,
            6.81199E-01,
            6.81199E-01,
            0.00000E+00,
            0.00000E+00,
            6.81199E-01,
            6.81199E-01,
            0.00000E+00,
            0.00000E+00,
            6.81199E-01,
            6.81199E-01,
            0.00000E+00,
            0.00000E+00,
            7.22071E-01,
            7.22071E-01,
            0.00000E+00,
            0.00000E+00,
            8.85559E-01,
            8.85559E-01,
            0.00000E+00,
            0.00000E+00,
            6.81199E-01,
            6.81199E-01,
            0.00000E+00,
            0.00000E+00,
            9.01907E-01,
            9.01907E-01,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            9.88808E-01,
            9.88808E-01,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            7.62943E-01,
            7.62943E-01,
            0.00000E+00,
            0.00000E+00,
            7.25903E-01,
            7.25903E-01,
            0.00000E+00,
            0.00000E+00,
            8.17439E-01,
            8.17439E-01,
            0.00000E+00,
            0.00000E+00,
            9.22480E-01,
            9.22480E-01,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            8.85559E-01,
            8.85559E-01,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            7.60802E-01,
            7.60802E-01,
            0.00000E+00,
            0.00000E+00,
            9.27877E-01,
            9.27877E-01,
            1.00000E+00,
            1.00000E+00,
            9.27877E-01,
            9.27877E-01,
            0.00000E+00,
            0.00000E+00,
            9.35617E-01,
            9.35617E-01,
            0.00000E+00,
            0.00000E+00,
            0.00000E+00,
            0.00000E+00,
            9.86239E-01,
            9.86239E-01,
            1.00000E+00,
            1.00000E+00,
            9.25560E-01,
            9.25560E-01,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            0.00000E+00,
            0.00000E+00,
            6.81199E-01,
            6.81199E-01,
            0.00000E+00,
            0.00000E+00,
            9.53678E-01,
            9.53678E-01,
            0.00000E+00,
            0.00000E+00,
            9.53678E-01,
            9.53678E-01,
            0.00000E+00,
            0.00000E+00,
            9.53678E-01,
            9.53678E-01,
            9.40159E-01,
            9.40159E-01,
            9.40159E-01,
            9.40159E-01,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            0.00000E+00,
            0.00000E+00,
            9.80178E-01,
            9.80178E-01,
            0.00000E+00,
            0.00000E+00,
            9.79451E-01,
            9.79451E-01,
            1.00000E+00,
            1.00000E+00,
            9.71363E-01,
            9.71363E-01,
            0.00000E+00,
            0.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            1.00000E+00,
            6.79780E-01,
            6.79780E-01
            ]
            power_vals = _time_sorter(msre_times, msre_rel_powers, times)
        power_vals = np.asarray(power_vals)
        if times[-1] > 24*3600:
            times = times/(24*3600)
            xlab = '[d]'
        elif times[-1] > 3600:
            times = times/(3600)
            xlab = '[h]'
        elif times[-1] > 60:
            times = times/(60)
            xlab = '[m]'
        else:
            xlab = '[s]'

        plt.step(times, power_vals/1e6, where='post')
        plt.xlabel(f'Time {xlab}')
        plt.ylabel('Power [MW]')
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
            vector_form_1 = True
        else:
            vector_form_1 = False
        if np.size(term2) > 1:
            vector_form_2 = True
        else:
            vector_form_2 = False

        for zi, z in enumerate(self.positions):
            if vector_form_1:
                incore_term = term1[zi]
            else:
                incore_term = term1

            if vector_form_2:
                excore_term = term2[zi]
            else:
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
            self.concs.append(self._format_spatial(1, 0))
            #self.concs.append(np.zeros(self.spacenodes))
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
            (len(self.reduced_times)+1, self.spacenodes, self.num_nucs), dtype=np.float64)
        for nuclide in range(self.num_nucs):
            result_mat[0, :, nuclide] = self.concs[nuclide]
        return result_mat

    def _incore_spatial_source(self) -> np.typing.NDArray[np.float64]:
        """
        Convert the source into a spatially resolved source term
        """
        source_shape: list = list()
        max_in_pos: float = self.run_params['frac_in'] * self.positions[-1]

        if self.run_params['flux_shape'] == 'flat':
            for i, pos in enumerate(self.positions):
                shape = 1
                source_shape.append(shape)

        elif self.run_params['flux_shape'] == 'sin':
            warnings.warn('Sinusoidal flux shape has flux scaling pre-incorporated')
            for i, pos in enumerate(self.positions):
                shape = np.sin(np.pi * pos / max_in_pos)
                if pos > max_in_pos:
                    shape = 0
                source_shape.append(shape)
                
        return np.asarray(source_shape)



    def _set_flow(self, t: float) -> None:
        """
        Set the flow over space and time by modifying flow_vec

        Parameters
        ----------
        t : float
            time
        """
        if self.run_params['flow_version'] == 'constant':
            self.flow_vec = self.base_flow
        elif self.run_params['flow_version'] == 'expdec':
            self.flow_vec = np.asarray(self.base_flow) * np.exp(-t/(self.run_params['final_time'] / 16))
        elif self.run_params['flow_version'] == 'expinc':
            self.flow_vec = np.asarray(self.base_flow) * np.exp(t/self.run_params['final_time'])
        elif self.run_params['flow_version'] == 'lindec':
            self.flow_vec = np.asarray(self.base_flow) * (-t/self.run_params['final_time'] + 1)
        return None

    def _avg_plot_flow(self) -> None:
        plot_vals = list()
        times = self.run_params['times']
        if self.run_params['flow_version'] == 'constant':
            for t in times:
                plot_vals.append(np.mean(self.base_flow))
        elif self.run_params['flow_version'] == 'expdec':
            for t in times:
                plot_vals.append(np.mean(self.base_flow) * np.exp(-t/(self.run_params['final_time'] / 16)))
        elif self.run_params['flow_version'] == 'lindec':
            for t in times:
                plot_vals.append(np.mean(self.base_flow) * (-t/self.run_params['final_time'] + 1))
        if times[-1] > 24*3600:
            times = times/(24*3600)
            xlab = '[d]'
        elif times[-1] > 3600:
            times = times/(3600)
            xlab = '[h]'
        elif times[-1] > 60:
            times = times/(60)
            xlab = '[m]'
        else:
            xlab = '[s]'

        plt.step(times, plot_vals, where='post')
        plt.xlabel(f'Time {xlab}')
        plt.ylabel('Flow Rate [cm/s]')
        plt.savefig('images/flow_hist.png')
        plt.close()

    def _update_sources(self, ti):
        """
        Update source terms based on concentrations

        Parameters
        ----------
        ti : int
            Current time index

        """
        for gain_nuc in range(self.num_nucs):
            fission_source = self.power[ti]/self.p0 * self.FYs[gain_nuc] * self.fission_shape # fiss/cc-s
            decay_source = np.zeros(len(self.concs[gain_nuc]))
            for loss_nuc in range(self.num_nucs):
                try:
                    frac = self.dec_fracs[(loss_nuc, gain_nuc)]
                    decay_source += (frac *
                                     self.concs[loss_nuc] *
                                     self.lams[loss_nuc])
                except KeyError:
                    continue
            scaling_factor = 1
            if self.run_params['solver_method'] == 'ODE' and self.run_params['scaled_flux']:
                scaling_factor = self.run_params['frac_in']
            incore_source = fission_source * scaling_factor + decay_source
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
            scaling_factor = 1 
            if self.run_params['solver_method'] == 'ODE':
                if self.run_params['scaled_flux']:
                    scaling_factor = self.run_params['frac_in']
                    if self.repr_loc == 'in':
                        repr_factor = scaling_factor
                    elif self.repr_loc == 'ex':
                        repr_factor = 1 - scaling_factor
                    losses = self.lams[nuclide] + self.power[ti]/self.p0 * self.loss_rates[nuclide] * scaling_factor * self.fission_shape + self.reprs[nuclide] * (repr_factor)
                else:
                    losses = self.lams[nuclide] + self.power[ti]/self.p0 * self.loss_rates[nuclide] * self.fission_shape + self.reprs[nuclide]
                cur_nuc_losses = self._format_spatial(losses, losses)
            else:
                incore_losses = (self.lams[nuclide] + 
                                self.power[ti]/self.p0 * self.loss_rates[nuclide] * self.fission_shape)
                excore_losses = self.lams[nuclide]
                if self.repr_loc == 'in':
                    incore_losses += self.reprs[nuclide]
                elif self.repr_loc == 'ex':
                    excore_losses += self.reprs[nuclide]
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
        loss = np.mean(self.mu[nuclide_index][0:self.transition_index])
        source = np.mean(self.S[nuclide_index][0:self.transition_index])

        

        conc = ((conc + source * self.dt) / (1 + loss * self.dt))


        return conc

    def _trim_result_matrix(self, res_mat):
        if res_mat.shape[0] > len(self.run_params['reduced_times']):
            res_mat = np.delete(res_mat, -1, axis=0)
        return res_mat
    
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
        advection_term = (conc[Jm1] - conc) / dz # First order upwind
        conc = ((conc + self.dt * (S_vec + self.flow_vec * advection_term)) / (1 + mu_vec * self.dt))

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
        self.scaling_factor = 1
        res_index = 1
        if self.run_params['scaled_flux']:
            self.scaling_factor = self.run_params['frac_in']
        for ti, t in enumerate(self.times[1:]):
            self._update_sources(ti)
            self._update_losses(ti)

            for nuclide in range(self.num_nucs):
                self.concs[nuclide] = self._external_ODE_no_step(
                    self.concs[nuclide], nuclide)
            
            if ti%self.run_params['time_mult'] == 0:
                ODE_result_mat = self._update_result_mat(ODE_result_mat, res_index)
                res_index += 1
        if self.run_params['time_mult'] > 1:
            ODE_result_mat = self._update_result_mat(ODE_result_mat, res_index)
        ODE_result_mat = self._trim_result_matrix(ODE_result_mat)
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
        res_index = 1

        if self.run_params['speed_factor_calc']:
            looped = False
            speeds = []
            t_use = 0
            for ti, t in enumerate(self.times[:-1]):
                self._set_flow(t)
                self._update_sources(ti)
                self._update_losses(ti)

                for nuclide in range(self.num_nucs):
                    self.concs[nuclide] = self._external_PDE_no_step(
                        self.concs[nuclide], nuclide)
                
                if np.isclose(self.concs[nuclide][-1], 1) and not looped:
                    speed = self.positions[-1] * (1  - self.run_params['frac_in']) / (t - t_use)
                    speeds.append(speed)
                    looped = True
                elif np.isclose(self.concs[nuclide][-1], 0) and looped:
                    t_use = t
                    looped = False
            print(f'Speeds : {speeds}')
            speed_factor = self.run_params['base_speed'] / np.mean(speeds[1:])
            print(f'Speed factor : {speed_factor}')
        else:
            for ti, t in enumerate(self.times[:-1]):
                self._set_flow(t)
                self._update_sources(ti)
                self._update_losses(ti)

                for nuclide in range(self.num_nucs):
                    self.concs[nuclide] = self._external_PDE_no_step(
                        self.concs[nuclide], nuclide)

                #if t in self.run_params['reduced_times']:
                if ti%self.run_params['time_mult'] == 0:
                    result_mat = self._update_result_mat(result_mat, res_index)
                    res_index += 1
            if self.run_params['time_mult'] > 1:
                result_mat = self._update_result_mat(result_mat, res_index)
            result_mat = self._trim_result_matrix(result_mat)
            self.result_mat = result_mat
            return result_mat
