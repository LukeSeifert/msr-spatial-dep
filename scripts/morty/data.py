import numpy as np


class DataHandler:
    def __init__(self, run_params):
        """
        This class handles the data for use in the PDE/ODE solvers.

        Parameters
        ----------
        run_params : dict
            key : str
                Name of run parameter

        """
        self.fissile_nuclide = run_params['fissile_nuclide']
        self.target_element = run_params['target_element']
        self.target_isobar = run_params['target_isobar']
        self.num_nucs = run_params['num_nuclides']
        scaling_factor = 1
        if run_params['scaled_flux'] and run_params['solver_method'] == 'ODE':
            scaling_factor = run_params['frac_in']
        self.flux = run_params['flux'] * scaling_factor
        self.power_W = run_params['power_W'] * scaling_factor
        self.nuclide_target = self.target_element + self.target_isobar
        data_gen_option = run_params['data_gen_option']
        self.run_params = run_params
        if data_gen_option == 'openmc':
            import openmc.data
            self.openmc_data_path = run_params['openmc_data_path']
            self.cur_target = self.nuclide_target
            self.temp = run_params['temperature']
            self.energy = run_params['neutron_energy']
            self.atomic_numbers = openmc.data.ATOMIC_NUMBER
            self.atomic_symbols = openmc.data.ATOMIC_SYMBOL
            self.endf_mt_total = [
                2,
                4,
                5,
                11,
                16,
                17,
                18,
                22,
                23,
                24,
                25,
                26,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                41,
                42,
                44,
                45,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117]
            self.chain_path = run_params['chain_path']
            self.data_params = self.openmc_data_gen()
        elif data_gen_option == 'hardcoded':
            self.data_params = self.hardcoded_data_gen()
        else:
            raise NotImplementedError('Invalid data_gen_option in DataHandler')
        return

    def _get_tot_xs(self, cur_target):
        """
        Gets total and fission cross section data

        Parameters
        ----------
        cur_target : str
            Nuclide targeted for information (e.g. Xe135)

        Returns
        -------
        micro_net_xs : float
            Microscopic total cross section
        macro_fiss_xs : float
            Macroscopic fission cross section for the fissile nuclide
        """
        import openmc.data
        has_data = True
        fiss_xs = 0
        fission_MT = 18
        try:
            hdf5_data_fissile = openmc.data.IncidentNeutron.from_hdf5(
                f'{self.openmc_data_path}{self.fissile_nuclide}.h5')
            hdf5_data_target = openmc.data.IncidentNeutron.from_hdf5(
                f'{self.openmc_data_path}{cur_target}.h5')
        except FileNotFoundError:
            print(f'{cur_target} does not have XS data')
            has_data = False
        net_xs = 0
        if has_data:
            try:
                fiss_xs_f = hdf5_data_fissile.reactions[fission_MT]._xs[self.temp]
                fiss_xs = fiss_xs_f(self.energy) * 1e-24
                self.run_params['fissile_atoms'] = self.power_W / \
                    (self.run_params['J_per_fiss'] * self.flux * fiss_xs)
                self.run_params['fissile_atom_dens_cc'] = self.run_params['fissile_atoms'] / \
                    self.run_params['net_cc_vol']
            except KeyError:
                print(f'No fission cross section')
                pass
            for MT in self.endf_mt_total:
                try:
                    f = hdf5_data_target.reactions[MT]._xs[self.temp]
                    net_xs += f(self.energy)
                except KeyError:
                    continue
        micro_net_xs = net_xs * 1e-24
        macro_fiss_xs = fiss_xs * self.run_params['fissile_atom_dens_cc']
        return micro_net_xs, macro_fiss_xs

    def openmc_data_gen(self):
        """
        Constructs dictionaries of data based on OpenMC data

        Returns
        -------
        data_params : dict
            key : str
                Name of data collection
            value : dict
                Dictionary of results
        """
        import openmc.deplete
        lams = {}
        FYs = {}
        loss_rates = {}
        decay_frac = {}
        tracked_nucs = {}
        tracked_element = self.target_element

        chain = openmc.deplete.Chain.from_xml(self.chain_path)
        yield_fracs = chain[self.fissile_nuclide].yield_data
        cur_target = f'{tracked_element}{self.target_isobar}'
        target_atomic_number = self.atomic_numbers[tracked_element]
        i = 0
        feeds = i - 1
        skip_feed = []
        _, fiss_xs = self._get_tot_xs(cur_target)
        while i < self.num_nucs:
            cur_target = f'{tracked_element}{self.target_isobar}'
            tracked_nucs[i] = cur_target
            lams[i] = openmc.data.decay_constant(cur_target)
            net_xs, _ = self._get_tot_xs(cur_target)
            FYs[i] = self.flux * fiss_xs * yield_fracs[self.energy][cur_target]
            loss_rates[i] = net_xs * self.flux

            decays = chain.nuclides[chain.nuclide_dict[cur_target]].decay_modes
            if i != 0 and len(decays) > 1:
                for dec in decays:
                    dec_type, target, ratio = dec
                    if self.target_isobar not in target:
                        continue
                    if target == prev_target:
                        feed_val = feeds
                        while feed_val in skip_feed:
                            feed_val += 1
                        decay_frac[(i, feed_val)] = ratio
                        continue
                    decay_frac[(i, feeds + 2)] = ratio

                    i += 1
                    if i >= self.num_nucs:
                        break
                    tracked_nucs[i] = target
                    net_xs, _ = self._get_tot_xs(cur_target)
                    loss_rates[i] = net_xs * self.flux
                    lams[i] = openmc.data.decay_constant(target)
                    FYs[i] = self.flux * fiss_xs * \
                        yield_fracs[self.energy][target]
                    if "_m" in target:
                        decay_frac[(i, feeds)] = 1
                        skip_feed.append(i)
            else:
                decay_frac[(i, feeds)] = 1
            i += 1
            feeds += 1
            target_atomic_number -= 1
            tracked_element = self.atomic_symbols[target_atomic_number]
            prev_target = cur_target

        data_params = {}
        data_params['lams'] = lams
        data_params['FYs'] = FYs
        data_params['dec_frac'] = decay_frac
        data_params['tracked_nucs'] = tracked_nucs
        data_params['loss_rates'] = loss_rates
        return data_params

    def hardcoded_data_gen(self, debug=False):
        """
        Hardocded data that does not require OpenMC

        Parameters
        ----------
        debug : bool
            Use simplified data for debugging purposes

        Returns
        -------
        data_params : dict
            key : str
                Name of data collection
            value : dict
                Dictionary of results
        """
        lams = {}
        FYs = {}
        loss_rates = {}
        decay_frac = {}
        tracked_nucs = {}
        tracked_element = self.target_element
        if self.nuclide_target == 'Xe135' and self.num_nucs <= 5:
            nuc_names = ['Xe135',
                         'I315',
                         'Xe135m',
                         'Te135',
                         'Sb135']
            half_life_data = [(15.29 * 3600),
                              (6.57 * 3600),
                              (9.14 * 3600),
                              19,
                              1.68]
            Ya = 0.00145764
            Yb = 0.0321618
            Yc = 0.0292737
            Yd_m1 = 0.0110156
            Yd = 0.000785125
            fiss_xs = 584.8972e-24
            self.run_params['fissile_atoms'] = self.power_W / \
                (self.run_params['J_per_fiss'] * self.flux * fiss_xs)
            self.run_params['fissile_atom_dens_cc'] = self.run_params['fissile_atoms'] / \
                self.run_params['net_cc_vol']
            fiss_macro_xs = fiss_xs * self.run_params['fissile_atom_dens_cc']
            yield_data = [Yd,
                          Yc,
                          Yd_m1,
                          Yb,
                          Ya]
            ng_I135 = 80.53724E-24
            ng_Xe135 = 2_666_886.8E-24
            net_xs_data = [ng_Xe135,
                           ng_I135,
                           0,
                           0,
                           0]
            decay_chain_path_data = [(0, -1),
                                     (1, 0),
                                     (1, 2),
                                     (2, 0),
                                     (3, 1),
                                     (4, 3)]
            decay_fracs_data = [1,
                                0.8349109,
                                0.1650891,
                                1,
                                1,
                                1]
        else:
            raise NotImplementedError(
                f'Hardcoded {self.nuclide_target} not available with {self.num_nucs} nuclides')

        for i in range(self.num_nucs):
            if debug:
                print('MODIFIED HARDCODED DATA FOR TESTING')
                tracked_nucs[i] = nuc_names[i]
                lams[i] = 1e-20
                FYs[i] = 1  # a/cc-s
                loss_rates[i] = 0
            tracked_nucs[i] = nuc_names[i]
            lams[i] = np.log(2) / half_life_data[i]
            FYs[i] = fiss_macro_xs * yield_data[i] * self.flux
            loss_rates[i] = net_xs_data[i] * self.flux
            for pathi, path in enumerate(decay_chain_path_data):
                if i == path[0]:
                    decay_frac[path] = decay_fracs_data[pathi]

        data_params = {}
        data_params['lams'] = lams
        data_params['FYs'] = FYs
        data_params['dec_frac'] = decay_frac
        data_params['tracked_nucs'] = tracked_nucs
        data_params['loss_rates'] = loss_rates
        return data_params
