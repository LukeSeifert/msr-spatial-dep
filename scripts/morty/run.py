import solvers
import data
import analysis
import plotter
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
            raise UserWarning(
                'Used value in data does not align with allowed values')
    return


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    plotting_params = {}
    plotting_params['plotting'] = True
    plotting_params['y_scale'] = 'log'
    plotting_params['gif'] = False
    plotting_params['parasitic_absorption'] = False
    plotting_params['image_directory'] = './images/'
    plotting_params['msre'] = False

    analysis_params = {}
    analysis_params['test_run'] = False
    test_name = 'Scaled Flux'
    analysis_params['PDE_ODE_compare'] = True
    analysis_params['nuclide_refinement'] = False
    analysis_params['spatial_refinement'] = False

    run_params = {}
    run_params['scaled_flux'] = True
    run_params['openmc_data_path'] = '/home/luke/projects/cross-section-libraries/nndc_hdf5/'
    run_params['temperature'] = '294K'
    run_params['neutron_energy'] = 0.0253
    run_params['chain_path'] = '../../data/chain_endfb71_pwr.xml'
    run_params['fissile_nuclide'] = 'U235'
    run_params['target_element'] = 'Xe' #'Nb'
    run_params['target_isobar'] = '135' #'95'
    run_params['spacenodes'] = 200
    run_params['num_nuclides'] = 5
    run_params['data_gen_option'] = 'openmc'
    run_params['final_time'] = 1.25*24*3600 + 100_000 #29_210_400 #1.25 * 24 * 3600 #5
    run_params['solver_method'] = 'ODE'
    run_params['flux'] = 6e12 # 2.9e12
    run_params['frac_in'] = 0.33 #0.272
    run_params['CFL_cond'] = 0.9
    #run_params['num_times'] = int(5e5)
    run_params['p0'] = 8e6
    run_params['power_version'] = 'step'
   # run_params['fissile_atom_dens_cc'] = 8.41e19
    run_params['reprocessing'] = {'Xe': 1/20,
                                  'I' : 1/20,
                                  'Te': 1/20,
                                  'Sb': 1/20}

    # https://www.osti.gov/servlets/purl/1488384
    run_params['residence_time'] = 8 # s
    #run_params['linear_flow_rate'] = 600 # cm/s
    run_params['linear_flow_rate'] = 21.75 #21.75 # cm/s
    #run_params['net_length'] = run_params['residence_time'] * 600 #608.06 cm
    run_params['net_length'] = 608.06
    #run_params['vol_flow_rate'] = 75708
    #run_params['fuel_fraction'] = 0.225
    #run_params['core_rad'] = 140.335 / 2
    run_params['net_cc_vol'] = 2_116_111
    run_params['J_per_fiss'] = 3.2e-11


    allowed_params = {}
    available_temperatures = ['294K']
    available_energies = [0.0253, 500_000, 14_000_000]
    available_data = ['openmc', 'hardcoded']
    available_methods = ['ODE', 'PDE']
    available_versions = ['constant', 'sin', 'neg_exp', 'msre', 'step']

    allowed_params['temperature'] = available_temperatures
    allowed_params['neutron_energy'] = available_energies
    allowed_params['data_gen_option'] = available_data
    allowed_params['solver_method'] = available_methods
    allowed_params['power_version'] = available_versions

    check_data(run_params, allowed_params)
    data_params = data.DataHandler(run_params).data_params
    #solvers.DiffEqSolvers(run_params, data_params, run=False)

    analyzer = analysis.AnalysisCollection(
        analysis_params, run_params, data_params)
    plotter_tool = plotter.PlotterCollection(
        plotting_params, run_params, data_params)

    if analysis_params['test_run']:
        print('-' * 50)
        data_dict = analyzer.test_run(name=test_name)
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])

    if analysis_params['PDE_ODE_compare']:
        print('-' * 50)
        data_dict = analyzer.ode_pde_compare()
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[0, -1,
                                                                 int(run_params['spacenodes']*run_params['frac_in'])])

    if analysis_params['nuclide_refinement']:
        print('-' * 50)
        data_dict = analyzer.nuclide_refinement(max_nuc=6)
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])

    if analysis_params['spatial_refinement']:
        print('-' * 50)
        spatial_nodes = [5, 10, 100, 200, 500, 1000]
        data_dict = analyzer.spatial_refinement(spatial_nodes)
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])
