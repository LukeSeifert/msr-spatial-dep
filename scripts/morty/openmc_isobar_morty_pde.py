import numpy as np
import matplotlib.pyplot as plt
from time import time
from morty_pde_ode_compare import FormatAssist
import os
import openmc
import openmc.data
import openmc.deplete


class IsobarSolve(FormatAssist):
    def __init__(self, nodes, z1, z2, nu1, nu2, lmbda, tf,
                 lams, FYs, dec_fracs, nucs, loss_rates,
                 vol1, vol2):
        """
        This class allows for the solve of a system of PDEs by
        solving each individually in a Jacobi-like manner.
        This approach will provide a more accurate result,
        as contributions from other nuclides in the isobar
        will be included.
        The next step up from this approach is to incorporate
        the spatial solve within the depletion solver itself.

        Parameters
        ----------
        nodes : int
            Number of spatial nodes
        z1 : float
            Position of in-core to ex-core transition
        z2 : float
            Position of ex-core to in-core transition
        nu1 : float
            Velocity in-core
        nu2 : float
            Velocity ex-core
        lmbda : float
            Time step times velocity divized by spatial mesh size
        tf : float
            Final time
        lams : dict
            key : str
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Decay constant for given nuclide
        FYs : dict
            key : string
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Fission yield for nuclice
        br_c_d : float
            Branching ratio for nuclide "c" to "d"
        br_dm1_d : float
            Branching ratio for nuclide "dm1" to "d"
        vol1 : float
            Volume of in-core region
        vol2 : float
            Volume of ex-core region
        losses : dict
            key : string
                Isobar nuclide indicator (a, b, c, d, or d_m1)
            value : float
                Loss term due to parasitic absorption or other similar terms

        """
        self.count = len(nucs)
        self.nodes = nodes
        self.z1 = z1
        self.zs = np.linspace(0, z1+z2, nodes)
        self.dt = lmbda * dz / nu1
        self.ts = np.arange(0, tf+self.dt, self.dt)
        self.nu_vec = self._format_spatial(nu1, nu2)
        self.lams = lams
        self.dec_fracs = dec_fracs
        self.FYs = FYs
        self.vol1 = vol1
        self.vol2 = vol2

        self.mu = {}
        for nuclide in range(self.count):
            incore_losses = lams[nuclide] + loss_rates[nuclide]
            excore_losses = lams[nuclide]
            cur_nuc_losses = self._format_spatial(incore_losses, excore_losses)
            self.mu[nuclide] = cur_nuc_losses

        self.S = {}
        self.S[self.count] = self._format_spatial((FYs[-1]/vol1), (0/vol2))


        return

    def _external_PDE_no_step(self, conc, isotope):
        """
        This function applies a single time step iteration of the PDE

        Parameters
        ----------
        conc : 1D vector
            Concentration over spatial nodes at previous time
        isotope : int
            Nuclide isobar indicator

        Returns
        -------
        conc : 1D vector
            Concentration over spatial nodes at current time
        """
        S_vec = self.S[isotope]
        mu_vec = self.mu[isotope]
        J = np.arange(0, self.nodes)
        Jm1 = np.roll(J,  1)
        dz = np.diff(self.zs)[0]

        conc_mult = 1 - mu_vec * self.dt
        add_source = S_vec * self.dt
        lmbda = (self.nu_vec * self.dt / dz)
        conc = add_source + conc_mult * conc + lmbda * (conc[Jm1] - conc)

        return conc

    def _initialize_result_mat(self, PDE=True):
        """
        Set up the 3D result matrix with the form
            time, space, nuclide
            with 5 nucldies in the isobar available

        Parameters
        ----------
        PDE : bool
            If using PDE True, False for ODE

        Returns
        -------
        result_mat : 3D matrix (2D if PDE False)
            Holds values over time, space, and nuclide (in that order)

        """
        if PDE:
            nodes = self.nodes
        else:
            nodes = 1
        result_mat = np.zeros((len(self.ts), nodes, self.count))
        self.concs = []
        for nuclide in range(self.count):
            self.concs[nuclide] = np.array([0] * nodes)
            result_mat[0, :, nuclide] = self.concs[nuclide]
        return result_mat

    def _update_sources(self, PDE=True):
        """
        Update source terms based on concentrations

        Parameters
        ----------
        PDE : bool
            True if PDE, False if ODE

        """
        if PDE:
            vector_form = True
        else:
            vector_form = False
        for gain_nuc in range(self.count):
            fission_source = self.FYs[gain_nuc]/self.vol1
            decay_source = np.asarray(self.concs[gain_nuc] * 0)
            for loss_nuc in range(self.count):
                frac = self.dec_fracs[(loss_nuc, gain_nuc)]
                decay_source += frac * self.concs[loss_nuc] * self.lams[loss_nuc]
            incore_source = fission_source + decay_source
            excore_source = decay_source
            cur_source = self._format_spatial(incore_source, excore_source,
                                              vector_form=vector_form)
            self.S[gain_nuc] = cur_source
        return

    def _update_result_mat(self, result_mat, ti):
        """
        Updates the result matrix with new concentrations

        Parameters
        ----------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        ti : int
            Current time index

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)
        """
        for nuclide in range(self.counts):
            result_mat[ti, :, nuclide] = self.concs[nuclide]
        return result_mat

    def serial_MORTY_solve(self):
        """
        Run MORTY, calculating the isobar concentrations in-core and ex-core
            for the given problem.

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)

        """
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(ts[:-1]):

            self._update_sources()

            for nuclide in range(self.count):
                self.concs[nuclide] = self._external_PDE_no_step(self.concs[nuclide], nuclide)

            result_mat = self._update_result_mat(result_mat, ti)

        return result_mat

    def _external_ODE_no_step(self, conc, isotope, t):
        """
        This function applies a single time step iteration of the ODE

        Parameters
        ----------
        conc : float
            Initial concentration
        isotope : string
            Nuclide isobar indicator (a, b, c, d, or d_m1)

        Returns
        -------
        conc : float
            Concentration at current time
        """

        conc = (conc * np.exp(-self.mu[isotope][0] * t) +
                self.S[isotope][0] / self.mu[isotope][0] *
                (1 - np.exp(-self.mu[isotope][0] * t)))

        return conc

    def ode_solve(self):
        """
        Solve the time dependent ODE

        """
        ODE_result_mat = self._initialize_result_mat(False)
        conc0_a = self.conc_a
        conc0_b = self.conc_b
        conc0_c = self.conc_c
        conc0_d = self.conc_d
        conc0_d_m1 = self.conc_d_m1

        for ti, t in enumerate(ts[:-1]):
            self._update_sources(False)

            self.conc_a = self._external_ODE_no_step(conc0_a, 'a', t)
            self.conc_b = self._external_ODE_no_step(conc0_b, 'b', t)
            self.conc_c = self._external_ODE_no_step(conc0_c, 'c', t)
            self.conc_d_m1 = self._external_ODE_no_step(conc0_d_m1, 'd_m1', t)
            self.conc_d = self._external_ODE_no_step(conc0_d, 'd', t)

            ODE_result_mat = self._update_result_mat(ODE_result_mat, ti)

        return ODE_result_mat

    def parallel_MORTY_solve(self):
        """
        Run MORTY, calculating the isobar concentrations in-core and ex-core
            for the given problem.

        Returns
        -------
        result_mat : 3D matrix
            Holds values over time, space, and nuclide (in that order)

        """
        import multiprocessing
        result_mat = self._initialize_result_mat()
        for ti, t in enumerate(ts[:-1]):
            # Parallelization is easy due to Jacobi appraoch
            #   Each isobar is independent from the others at each iteration

            self._update_sources()

            with multiprocessing.Pool() as pool:
                res_list = pool.starmap(self._external_PDE_no_step,
                                        [(self.conc_a, 'a'),
                                         (self.conc_b, 'b'),
                                         (self.conc_c, 'c'),
                                         (self.conc_d_m1, 'd_m1'),
                                         (self.conc_d, 'd')])
            self.conc_a = res_list[0]
            self.conc_b = res_list[1]
            self.conc_c = res_list[2]
            self.conc_d_m1 = res_list[3]
            self.cond_d = res_list[4]

            result_mat = self._update_result_mat(result_mat, ti)

        return result_mat

def get_tot_xs(cur_target, endf_mt_total, temp):
    has_data = True
    try:
        hdf5_data = openmc.data.IncidentNeutron.from_hdf5(f'/root/nndc_hdf5/{cur_target}.h5')
    except FileNotFoundError:
        print(f'{cur_target} does not have XS data')
        has_data = False
    net_xs = 0
    if has_data:
        for MT in endf_mt_total:
            try:
                f = hdf5_data.reactions[MT]._xs[temp]
                net_xs += f(selected_energy)
            except KeyError:
                continue
    net_xs = net_xs * 1e-24
    return net_xs


def build_data(chain_path, fissile_nuclide, target_element, target_isobar,
               number_tracked, temp, selected_energy, flux):
    """
    Constructs required dictionaries
    """
    atomic_numbers = openmc.data.ATOMIC_NUMBER
    atomic_symbols = openmc.data.ATOMIC_SYMBOL
    endf_mt_total = [2, 4, 5, 11, 16, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30,
                     31, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45, 102, 103,
                     104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                     114, 115, 116, 117]
    lams = {}
    FYs = {}  # atoms/s = fissions/J * J/s * yield_fraction
    loss_rates = {}
    decay_frac = {}
    tracked_nucs = {}
    chain = openmc.deplete.Chain.from_xml(chain_path)
    yield_fracs = chain[fissile_nuclide].yield_data
    cur_target = f'{target_element}{target_isobar}'
    target_atomic_number = atomic_numbers[target_element]
    i = 0
    feeds = i - 1
    while i < number_tracked:
        cur_target = f'{target_element}{target_isobar}'
        tracked_nucs[i] = cur_target
        lams[i] = openmc.data.decay_constant(cur_target)
        FYs[i] = PC * P * yield_fracs[selected_energy][cur_target]
        net_xs = get_tot_xs(cur_target, endf_mt_total, temp)
        loss_rates[i] = net_xs * flux

        # If doesn't branch to metastable that decays into one of the isobars
        decays = chain.nuclides[chain.nuclide_dict[cur_target]].decay_modes
        if i != 0 and len(decays) > 1:
            for dec in decays:
                dec_type, target, ratio = dec
                if target == prev_target:
                    decay_frac[(i, feeds)] = ratio
                    continue
                feeds += 2
                #lams[(i, feeds)] = openmc.data.decay_constant(cur_target)
                #FYs[(i, feeds)] = PC * P * yield_fracs[selected_energy][cur_target]
                decay_frac[(i, feeds)] = ratio

                feeds -= 2
                i += 1
                if i >= number_tracked:
                    break
                tracked_nucs[i] = target
                net_xs = get_tot_xs(target, endf_mt_total, temp)
                loss_rates[i] = net_xs * flux
                lams[i] = openmc.data.decay_constant(target)
                FYs[i] = PC * P * yield_fracs[selected_energy][target]
                decay_frac[(i, feeds)] = 1
        else:
            decay_frac[(i, feeds)] = 1
        i += 1
        feeds += 1
        target_atomic_number -= 1
        target_element = atomic_symbols[target_atomic_number]
        prev_target = cur_target
    print(lams)
    print(FYs)
    print(decay_frac)
    print(loss_rates)
    print(tracked_nucs)
    return lams, FYs, decay_frac, tracked_nucs, loss_rates


if __name__ == '__main__':
    # Test this module using MSRE 135 isobar
    parallel = False
    gif = False
    ode = True
    scaled_flux = True
    savedir = './images'
    chain_file = '../../data/chain_endfb71_pwr.xml'
    number_tracked = 5
    tf = 1.25 * 24 * 3600
    fissile_nuclide = 'U235'
    target_isobar = '135'
    target_element = 'Xe'
    spacenodes = 100
    flux = 6E12
    selected_energy = 0.0253
    selected_temp = '294K'
    available_temperatures = ['294K']
    available_energies = [0.0253, 500_000, 14_000_000]
    PC = 1 / (3.2e-11)
    P = 8e6  # 8MW
    
    lams, FYs, dec_fracs, nucs, loss_rates = build_data(chain_file,
                                                           fissile_nuclide,
                                                           target_element,
                                                           target_isobar,
                                                           number_tracked,
                                                           selected_temp,
                                                           selected_energy,
                                                           flux)

    input('Paused')
    L = 608.06  # 824.24
    V = 2116111
    frac_in = 0.33
    frac_out = 0.67
    core_outlet_node = int(spacenodes * frac_in)
    z1 = frac_in * L
    z2 = frac_out * L
    vol1 = frac_in * V
    vol2 = frac_out * V
    core_diameter = 140.335
    fuel_frac = 0.225
    xsarea = fuel_frac * (np.pi * (core_diameter/2)**2)
    nu = 75708 / xsarea
    nu1 = nu
    nu2 = nu
    loss_core = 6e12 * 2666886.8E-24
    dz = np.diff(np.linspace(0, z1+z2, spacenodes))[0]
    lmbda = 0.9
    dt = lmbda * dz / nu1
    ts = np.arange(0, tf+dt, dt)
    print(f'Number of iterations: {len(ts)}')
    #isotopea = 'Sb135'
    #isotopeb = 'Te135'
    #isotopec = 'I135'
    #isotoped_m1 = 'Xe135_m1'
    #isotoped = 'Xe135'
    #br_c_d = 0.8349109
    #br_dm1_d = 0.997
    #lams = {}
    #lams['a'] = np.log(2) / 1.68
    #lams['b'] = np.log(2) / 19
    #lams['c'] = np.log(2) / (6.57*3600)
    #lams['d'] = np.log(2) / (15.29*3600)
    #lams['d_m1'] = np.log(2) / (9.14*3600)

    zs = np.linspace(0, z1+z2, spacenodes)

    # Yields from ENDF OpenMC thermal data
    #Ya = 0.00145764
    #Yb = 0.0321618
    #Yc = 0.0292737
    #Yd_m1 = 0.0110156
    #Yd = 0.000785125
    #FYs = {}  # atoms/s = fissions/J * J/s * yield_fraction
    #FYs['a'] = PC * P * Ya
    #FYs['b'] = PC * P * Yb
    #FYs['c'] = PC * P * Yc
    #FYs['d'] = PC * P * Yd
    #FYs['d_m1'] = PC * P * Yd_m1
    #phi_th = 6E12
    #losses = {}
    #losses['1a'] = 0
    #losses['2a'] = 0
    #losses['1b'] = 0
    #losses['2b'] = 0
    #losses['2c'] = 0
    #losses['2d'] = 0
    #ng_I135 = 80.53724E-24
    #ng_Xe135 = 2_666_886.8E-24
    #ng_Xe135_m1 = 0  # 10_187_238E-24
    #losses['1c'] = phi_th * ng_I135
    #losses['1d'] = phi_th * ng_Xe135
    #losses['1d_m1'] = phi_th * ng_Xe135_m1

    start = time()
    # lams, FYs, dec_fracs, nucs, loss_rates
    solver = IsobarSolve(spacenodes, z1, z2, nu1, nu2, lmbda, tf,
                         lams, FYs, dec_fracs, nucs, loss_rates,
                         vol1, vol2)
    if parallel:
        result_mat = solver.parallel_MORTY_solve()
    else:
        result_mat = solver.serial_MORTY_solve()
    if ode:
        if scaled_flux:
            P = 8e6 * frac_in
            FYs['a'] = PC * P * Ya
            FYs['b'] = PC * P * Yb
            FYs['c'] = PC * P * Yc
            FYs['d'] = PC * P * Yd
            FYs['d_m1'] = PC * P * Yd_m1
            phi_th = 6E12 * frac_in
            losses['1c'] = phi_th * ng_I135
            losses['1d'] = phi_th * ng_Xe135
            losses['1d_m1'] = phi_th * ng_Xe135_m1
            solver = IsobarSolve(spacenodes, z1, z2, nu1, nu2, lmbda, tf,
                                 lams, FYs, br_c_d, br_dm1_d, vol1,
                                 vol2, losses)

        ode_result_mat = solver.ode_solve()
    end = time()
    print(f'Time taken : {round(end-start)}s')

    # Plotting
    savedir = './images'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    if tf > 3600 * 24:
        ts = ts / (3600*24)
        units = 'd'
    else:
        units = 's'
    labels = [isotopea, isotopeb, isotopec, isotoped_m1, isotoped]
    for i, iso in enumerate(labels):
        plt.plot(ts[:-2], result_mat[0:-2, core_outlet_node, i],
                 label=f'{iso} Exiting Core')
        plt.plot(ts[:-2], result_mat[0:-2, -1, i],
                 label=f'{iso} Entering Core')
        if ode:
            plt.plot(ts[:-2], ode_result_mat[0:-2, 0, i],
                     label=f'{iso} ODE', linestyle='--')

        plt.xlabel(f'Time [{units}]')
        plt.ylabel('Concentration [at/cc]')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{savedir}/nuc_{iso}_conc_time.png')
        plt.close()

    if ode:
        for i, iso in enumerate(labels):
            print('-' * 50)
            print(f'{iso} atom densities')
            PDE_val_core_inlet = result_mat[-2, 0, i]
            print(f'PDE core inlet: {PDE_val_core_inlet}')
            ODE_val = ode_result_mat[-2, 0, i]
            print(f'ODE {ODE_val}')
            pcnt_diff = ((PDE_val_core_inlet - ODE_val) /
                         (PDE_val_core_inlet) * 100)
            print(f'{iso} PDE/ODE diff: {round(pcnt_diff, 3)}%')

    # Gif
    if gif:
        print(f'Estimated time to gif completion: {round(0.08 * len(ts))} s')
        start = time()
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        max_conc = np.max(result_mat[0:-2, :, :])

        def update(frame):
            ax.clear()
            plt.xlabel('Space [cm]')
            plt.vlines(z1, 0, 1e1 * max_conc, color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.ylim((1e-5 * max_conc, 1e1 * max_conc))
            plt.yscale('log')

            for i, iso in enumerate(labels):
                ax.plot(zs, result_mat[frame, :, i],
                        label=f'{iso}', marker='.')
            ax.set_title(f'Time: {round(frame*dt, 4)} s')
            plt.legend()
        animation = FuncAnimation(fig, update, frames=len(ts), interval=1)
        animation.save(f'{savedir}/isobar_evolution.gif', writer='pillow')
        plt.close()
        print(f'Gif took {time() - start} s')
