import numpy as np
import matplotlib.pyplot as plt
from time import time
import os


class FormatAssist:
    """
    Mixin class

    """

    def _format_spatial(self, term1, term2, vector_form=False):
        """
        Distribute term 1 in < z1, and term 2 in > z1.
        Returns a list of terms corresponding to each z.
        If term1 and term2 are vectors, use `vector_form=True`

        Parameters
        ----------
        term1 : float
            Term in the in-core region
        term2 : float
            Term in the ex-core region
        vector_form : bool
            Terms are given as spatial vectors rather than floats

        Returns
        -------
        return_list : 1D numpy array
            Spatial distribution of values at each point

        """
        return_list = []
        for zi, z in enumerate(self.zs):
            if z <= self.z1:
                if not vector_form:
                    return_list.append(term1)
                else:
                    return_list.append(term1[zi])
            elif z > self.z1:
                if not vector_form:
                    return_list.append(term2)
                else:
                    return_list.append(term2[zi])
        return np.asarray(return_list)


class DiffEqComp(FormatAssist):
    def __init__(self, mu, S, isotope, nu1, nu2, z1, z2,
                 nodes, tf, dt, initial_guess, lmbda):
        """
        This class generates results for comparing differential equations.
        Particularly, this class compares ODE depletion in time to
        PDE depletion in 1D space and time.

        Parameters
        ----------
        mu : dict
            key : string
                Name of isotope
            value : List of float
                Value of loss in-core and value ex-core
        S : dict
            key : string
                Name of isotope
            value : List of float
                Value of source in-core and value ex-core
        isotope : string
            Name of isotope to model
        nu1 : float
            Linear flow rate in-core
        nu2 : float
            Linear flow rate ex-core
        z1 : float
            Linear height of in-core region
        z2 : float
            Linear height of ex-core region
        nodes : int
            Number of spatial nodes (split between in- and ex-core)
        tf : float
            End time value in seconds
        dt : float
            Time step value in seconds
        initial_guess : List of floats
            Initial concentration in-core and ex-core
        lmbda : float
            Numerical stability value (must be <1);
            Time step times flow rate divided by spatial step

        """
        self.mu1, self.mu2 = mu[isotope]
        self.S1, self.S2 = S[isotope]
        self.nu1, self.nu2 = nu1, nu2
        self.nodes = nodes
        self.zs = np.linspace(0, z1+z2, nodes)
        self.dt = dt
        self.tf = tf
        self.ts = np.arange(0, tf+dt, dt)
        self.z1 = z1
        self.init_conc = self._format_spatial(initial_guess[0],
                                              initial_guess[1])
        self.lmbda = lmbda
        return

    def _format_spatial(self, term1, term2, vector_form=False):
        """
        Distribute term 1 in < z1, and term 2 in > z1.
        Returns a list of terms corresponding to each z.
        If term1 and term2 are vectors, use `vector_form=True`

        Parameters
        ----------
        term1 : float
            Term in the in-core region
        term2 : float
            Term in the ex-core region
        vector_form : bool
            Terms are given as spatial vectors rather than floats

        Returns
        -------
        return_list : 1D numpy array
            Spatial distribution of values at each point

        """
        return_list = []
        for zi, z in enumerate(self.zs):
            if z < self.z1:
                if not vector_form:
                    return_list.append(term1)
                else:
                    return_list.append(term1[zi])
            elif z > self.z1:
                if not vector_form:
                    return_list.append(term2)
                else:
                    return_list.append(term2[zi])
            else:
                if not vector_form:
                    return_list.append(0.5 * (term1 + term2))
                else:
                    return_list.append(0.5 * (term1[zi] + term2[zi]))
        return np.asarray(return_list)

    def _step_source(self):
        """
        The source and loss terms are treated as a step function for
        the ex-core and in-core regions, each having a single value.

        Returns
        -------
        S_vec : 1D vector
            Source terms over space
        nu_vec : 1D vector
            Flow rate terms over space
        mu_vec : 1D vector
            Loss terms over space

        """
        S_vec = self._format_spatial(self.S1, self.S2, False)
        nu_vec = self._format_spatial(self.nu1, self.nu2, False)
        mu_vec = self._format_spatial(self.mu1, self.mu2, False)
        return S_vec, nu_vec, mu_vec

    def _vary_source(self):
        """
        The source and loss terms are treated as a sinusoidal
        profile in the in-core region rather than as a constant.

        Returns
        -------
        S_vec : 1D vector
            Source terms over space
        nu_vec : 1D vector
            Flow rate terms over space
        mu_vec : 1D vector
            Loss terms over space

        """
        core_mod = np.pi/2 * np.sin(self.zs * np.pi / self.z1)
        S_vec = self._format_spatial(self.S1*core_mod, self.S2,
                                     vector_form=True)
        nu_vec = self._format_spatial(self.nu1, self.nu2)
        mu_vec = self._format_spatial(self.mu1*core_mod, self.mu2,
                                      vector_form=True)
        return S_vec, nu_vec, mu_vec

    def fd_PDE(self, spatial_vary):
        """
        Solves the PDE using finite difference explicit in time and
        upwind (backwards) in space.

        Parameters
        ----------
        spatial_vary : bool
            True means the source and loss terms will have a sinusoidal shape
            in the in-core region.

        Returns
        -------
        zs : 1D vector
            Spatial nodes
        ts : 1D vector
            Time nodes
        N_z_t : 2D matrix
            Each row represents the spatial concentration
            at a given point in time

        """
        conc = self.init_conc
        if not spatial_vary:
            S_vec, nu_vec, mu_vec = self._step_source()
        else:
            S_vec, nu_vec, mu_vec = self._vary_source()
        J = np.arange(0, self.nodes)
        Jm1 = np.roll(J,  1)

        N_z_t = np.zeros((len(self.ts)+1, self.nodes))
        N_z_t[0, :] = conc
        conc_mult = 1 - mu_vec * self.dt
        add_source = S_vec * self.dt
        for ti, t in enumerate(self.ts):
            conc = (add_source + conc_mult * conc
                    + self.lmbda * (conc[Jm1] - conc))
            N_z_t[ti+1, :] = conc

        return self.zs, self.ts, N_z_t

    def ODE(self):
        """
        Solves the ODE form of depletion (no neutronic updates)

        Returns
        -------
        conc : list of 1D vectors
            Each item in the list is the spatial concentration vector

        """
        conc = []
        for t in ts:
            current = (self.init_conc[0] * np.exp(-self.mu1 * t) +
                       self.S1/self.mu1 * (1 - np.exp(-self.mu1 * t)))
            conc.append(current)
        print(conc)
        return conc


if __name__ == '__main__':
    no_space_ODE = True
    spatial_source_variation = False
    gif = False
    scaled_flux = False
    tf = 3600 #100  # 324_000
    nodes = 40

    # MSRE data
    # https://www.tandfonline.com/doi/epdf/10.1080/00295450.2021.1943122 [1]
    # https://github.com/openmsr/msre/blob/master/core/docs/msrecore.pdf [2]
    # https://link.springer.com/article/10.1007/s10967-022-08535-3 [3]
    incore_frac = 1#0.33
    excore_frac = 0#0.67
    z1 = 200.66  # [2] #272 #[1]
    z2 = (z1 / incore_frac) * excore_frac

    # 25,233 cm3/s # [1]
    # 75,708 cm3/s # [2]
    core_diameter = 140.335  # [2]
    fuel_frac = 0.225  # [2]
    xsarea = fuel_frac * (np.pi * (core_diameter/2)**2)
    nu = 75708 / xsarea
    nu1 = nu  # 66666.66 [3]
    nu2 = nu  # 66666.66 [3]
    loss_core = 6e12 * 2666886.8E-24
    isotope = 'test'
    mu = {'test': [2.1065742176025568e-05 + loss_core, 2.1065742176025568e-05]}
    S = {'test': [24568909090.909092, 0]}
    initial_guess = [0, 0]
    nz = 20

    dz = np.diff(np.linspace(0, z1+z2, nz))[0]

    lmbda = 0.9
    dt = lmbda * dz / nu1
    Comp = DiffEqComp(mu, S, isotope, nu1, nu2, z1, z2,
                      nodes, tf, dt, initial_guess, lmbda)
    print(f'Spatial nodes: {nodes}')
    ts = np.arange(0, tf+dt, dt)
    print(f'Temporal nodes: {len(ts)}')
    print(f'Net data: {nodes * len(ts)}')

    start = time()
    zs, ts, concs = Comp.fd_PDE(False)
    if spatial_source_variation:
        _, _, concs_vary = Comp.fd_PDE(True)

    if no_space_ODE:
        if scaled_flux:
            mu = {'test': [2.1065742176025568e-05 + loss_core * incore_frac,
                           2.1065742176025568e-05]}
            S = {'test': [24568909090.909092 * incore_frac, 0]}
            Comp = DiffEqComp(mu, S, isotope, nu1, nu2, z1, z2,
                              nodes, tf, dt, initial_guess, lmbda)
        conc_no_space = Comp.ODE()
    end = time()
    print(f'Time taken: {round(end-start)} s')

    # Plotting
    savedir = './images'
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    plt.plot(zs, concs[0, :], label='Initial')
    if spatial_source_variation:
        plt.plot(zs, concs_vary[0, :], label='Initial (Spatial Source)',
                 linestyle='-.', color='b')
    if no_space_ODE:
        plt.hlines(conc_no_space[0], zs[0], zs[-1],
                   label='Initial No Spatial Component', linestyle='--')
        plt.vlines(z1, 0, np.max(conc_no_space), label='Core Outlet',
                   color='black')
    else:
        plt.vlines(z1, 0, np.max(concs), color='black')
    plt.plot(zs, concs[int(len(ts)/2), :], label='Intermediate')
    if spatial_source_variation:
        plt.plot(zs, concs_vary[int(len(ts)/2), :],
                 label='Intermediate (Spatial Source)',
                 linestyle='-.', color='orange')
    if no_space_ODE:
        plt.hlines(conc_no_space[int(len(ts)/2)], zs[0], zs[-1],
                   label='Intermediate No Spatial Component',
                   linestyle='--', color='orange')
    plt.plot(zs, concs[-1, :], label='Final')
    if spatial_source_variation:
        plt.plot(zs, concs_vary[-1, :], label='Final (Spatial Source)',
                 linestyle='-.', color='g')
    if no_space_ODE:
        plt.hlines(conc_no_space[-1], zs[0], zs[-1],
                   label='Final No Spatial Component',
                   linestyle='--', color='g')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Space [cm]')
    plt.ylabel('Conc [at/cc]')
    plt.savefig(f'{savedir}/PDE_ODE_space.png')
    plt.close()

    plt.plot(ts, concs[:, int(len(zs)/3)][:-1], label='Core Outlet')
    plt.plot(ts, concs[:, -1][:-1], label='Excore Outlet')
    if spatial_source_variation:
        plt.plot(ts, concs_vary[:, int(len(zs)/3)][:-1],
                 label='Core Outlet (Spatial Source)',
                 linestyle='-.', color='b')
        plt.plot(ts, concs_vary[:, -1][:-1],
                 label='Excore Outlet (Spatial Source)',
                 linestyle='-.', color='orange')

    if no_space_ODE:
        plt.plot(ts, conc_no_space, label='No Spatial Component')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('Conc [at/cc]')
    plt.savefig(f'{savedir}/PDE_ODE_time.png')
    plt.close()

    # Core inlet differences
    if spatial_source_variation:
        fnl_pct_diff = (np.abs(concs[-1, -1] - concs_vary[-1, -1])
                        / (2 * (concs[-1, -1] + concs_vary[-1, -1])) * 100)
        print(f'Percent difference core inlet spatial source: {fnl_pct_diff}%')
        avg_pcnt_diff = (np.abs(np.mean(concs[-1, :])
                                - np.mean(concs_vary[-1, :]))
                         / (2 * (np.mean(concs[-1, :])
                                 + np.mean(concs_vary[-1, :]))) * 100)
        print(f'Percent difference average spatial source: {avg_pcnt_diff}%')

    if no_space_ODE:
        final_pcnt_diff = (np.abs(concs[-1, -1] - conc_no_space[-1])
                           / (2 * (concs[-1, -1] + conc_no_space[-1])) * 100)
        print(f'Percent difference core inlet ODE: {final_pcnt_diff}%')
        mean_conc = np.mean(concs[-1, :])
        mean_ns_conc = conc_no_space[-1]
        avg_pcnt_diff = (np.abs(mean_conc - mean_ns_conc)
                         / (2 * (mean_conc + mean_ns_conc)) * 100)
        print(f'Percent difference average ODE: {avg_pcnt_diff}%')

    # Gif
    if gif:
        print(f'Estimated time to gif completion: {round(0.03 * len(ts))} s')
        start = time()
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        max_val = np.max(concs[:, :])

        def update(frame):
            ax.clear()
            plt.xlabel('Space [cm]')
            plt.vlines(z1, 0, 1e1 * max_val, color='black')
            plt.ylabel('Concentration [at/cc]')
            plt.yscale('log')
            plt.ylim((1e-5 * max_val, 1e1 * max_val))

            ax.plot(zs, concs[frame, :], marker='.')
            ax.set_title(f'Time: {round(frame*dt, 4)} s')
        animation = FuncAnimation(fig, update, frames=len(ts), interval=1)
        animation.save(f'{savedir}/evolution.gif', writer='pillow')
        plt.close()
        print(f'Gif took {time() - start} s')
