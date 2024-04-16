def check_data(run_params, allowed_params):
    """
    Check that each item in run_params is in the allowed
        parameters
    
    """
    for key in allowed_params.keys():
        used_value = run_params[key]
        allowed_values = allowed_params[key]
        if used_value not in allowed_values:
            print(f'{key=}\n{used_value=}\n{allowed_values=}')
            raise UserWarning('Used value in data does not align with allowed values')
    return



if __name__ == '__main__':
    run_params = {}
    run_params['openmc_data_path'] = '/root/nndc_hdf5/'
    run_params['temperature'] = '294K'
    run_params['neutron_energy'] = 0.0253
    run_params['chain_path'] = '../../data/chain_endfb71_pwr.xml'
    run_params['fissile_nuclide'] = 'U235'
    run_params['target_element'] = 'Xe'
    run_params['target_isobar'] = '135'
    run_params['spacenodes'] = 200
    run_params['num_nuclides'] = 1
    run_params['data_gen_option'] = 'openmc'
    run_params['final_time'] = 100

    allowed_params = {}
    available_temperatures = ['294K']
    available_energies = [0.0253, 500_000, 14_000_000]
    available_data = ['openmc', 'hardcoded']

    allowed_params['temperature'] = available_temperatures
    allowed_params['neutron_energy'] = available_energies
    allowed_params['data_gen_option'] = available_data






    check_data(run_params, allowed_params)