import solvers
import data
import analysis
import numpy as np

def check_data(run_params, allowed_params):
    """
    Check that each item in run_params is in the allowed
        parameters

    Parameters
    ----------
    run_params : dict
        key : str
            Name of run parameter
    allowed_params : dict
        key : str
            Name of allowed parameter
    
    """
    for key in allowed_params.keys():
        used_value = run_params[key]
        allowed_values = allowed_params[key]
        if used_value not in allowed_values:
            print(f'{key=}\n{used_value=}\n{allowed_values=}')
            raise UserWarning('Used value in data does not align with allowed values')
    return



if __name__ == '__main__':
    analysis_params = {}
    analysis_params['test_run'] = False
    analysis_params['PDE_ODE_compare'] = True

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
    run_params['solver_method'] = 'PDE'
    run_params['flux'] = 6e12
    run_params['net_length'] = 608.06
    run_params['frac_in'] = 0.33
    run_params['frac_out'] = 1 - run_params['frac_in']
    run_params['core_outlet']   = (run_params['net_length'] * run_params['frac_out'])
    run_params['excore_outlet'] = run_params['core_outlet'] + (run_params['net_length'] * run_params['frac_in'])
    run_params['CFL_cond'] = 0.9
    vol_flow_rate = 75708
    fuel_fraction = 0.225
    core_rad = 140.335/2
    net_cc_vol = 2_116_111


    run_params['incore_volume'] = net_cc_vol * run_params['frac_in']
    linear_flow_rate = (vol_flow_rate / (fuel_fraction * np.pi * (core_rad)**2))
    run_params['incore_flowrate'] = linear_flow_rate
    run_params['excore_flowrate'] = linear_flow_rate
    max_flowrate = max(run_params['incore_flowrate'], run_params['excore_flowrate'])
    run_params['dz'] = np.diff(np.linspace(0, run_params['excore_outlet'], run_params['spacenodes']))[0]
    run_params['positions'] = np.linspace(0, run_params['excore_outlet'], run_params['spacenodes'])
    run_params['dt'] = run_params['CFL_cond'] * run_params['dz'] / max_flowrate
    run_params['times'] = np.arange(0, run_params['final_time']+run_params['dt'], run_params['dt'])


    allowed_params = {}
    available_temperatures = ['294K']
    available_energies = [0.0253, 500_000, 14_000_000]
    available_data = ['openmc', 'hardcoded']
    available_methods = ['PDE', 'ODE']

    allowed_params['temperature'] = available_temperatures
    allowed_params['neutron_energy'] = available_energies
    allowed_params['data_gen_option'] = available_data
    allowed_params['solver_method'] = available_methods


    check_data(run_params, allowed_params)
    data_params = data.DataHandler(run_params).data_params

    analyzer = analysis.AnalysisCollection(analysis_params, run_params, data_params)
    if analysis_params['test_run']:
        result_matrix = solvers.DiffEqSolvers(run_params, data_params).result_mat
    
    if analysis_params['PDE_ODE_compare']:
        data_dict = analyzer.ode_pde_compare()