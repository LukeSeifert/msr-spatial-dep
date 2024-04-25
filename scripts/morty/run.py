import solvers
import data
import analysis
import numpy as np
import plotter

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
    plotting = True
    image_directory = './images/'
    analysis_params = {}
    analysis_params['test_run'] = False
    analysis_params['PDE_ODE_compare'] = True
    analysis_params['nuclide_refinement'] = False
    analysis_params['spatial_refinement'] = False

    run_params = {}
    run_params['openmc_data_path'] = '/root/nndc_hdf5/'
    run_params['temperature'] = '294K'
    run_params['neutron_energy'] = 0.0253
    run_params['chain_path'] = '../../data/chain_endfb71_pwr.xml'
    run_params['fissile_nuclide'] = 'U235'
    run_params['target_element'] = 'Xe'
    run_params['target_isobar'] = '135'
    run_params['spacenodes'] = 200
    run_params['num_nuclides'] = 2
    run_params['data_gen_option'] = 'openmc'
    run_params['final_time'] = 3600
    run_params['solver_method'] = 'PDE'
    run_params['flux'] = 6e12
    run_params['net_length'] = 608.06
    run_params['frac_in'] = 0.33
    run_params['CFL_cond'] = 0.9
    run_params['power_W'] = 8e6

    run_params['vol_flow_rate'] = 75708
    run_params['fuel_fraction'] = 0.225
    run_params['core_rad'] = 140.335/2
    run_params['net_cc_vol'] = 2_116_111
    run_params['J_per_fiss'] = 3.2e-11



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
    solvers.DiffEqSolvers(run_params, data_params, run=False)

    analyzer = analysis.AnalysisCollection(analysis_params, run_params, data_params)
    plotter_tool = plotter.PlotterCollection(image_directory)

    if analysis_params['test_run']:
        print('-'*50)
        result_matrix = solvers.DiffEqSolvers(run_params, data_params).result_mat
        print(result_matrix)
    
    if analysis_params['PDE_ODE_compare']:
        print('-'*50)
        data_dict = analyzer.ode_pde_compare()
        if plotting:
            plotter_tool.plot_time(data_dict)
            core_outlet_node = int(run_params['spacenodes'] * run_params['frac_in'])
            plotter_tool.plot_time(data_dict, core_outlet_node)
    
    if analysis_params['nuclide_refinement']:
        print('-'*50)
        data_dict = analyzer.nuclide_refinement(max_nuc=5)
        if plotting:
            plotter_tool.plot_time(data_dict)

    if analysis_params['spatial_refinement']:
        print('-'*50)
        spatial_nodes = [2, 5, 10, 100, 200, 500]
        data_dict = analyzer.spatial_refinement(spatial_nodes)
        if plotting:
            plotter_tool.plot_time(data_dict)
