import solvers
import data
import analysis
import plotter
import numpy as np
from time import time


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
    start = time()
    np.set_printoptions(threshold=np.inf)
    plotting_params = {}
    plotting_params['plotting'] = True
    plotting_params['y_scale'] = 'linear'
    plotting_params['gif'] = False
    plotting_params['parasitic_absorption'] = False
    plotting_params['image_directory'] = './images/'
    plotting_params['msre'] = True
    plotting_params['surf_plot'] = False

    analysis_params = {}
    analysis_params['test_run'] = False
    test_name = 'Scaled Flux'
    analysis_params['PDE_ODE_compare'] = True
    analysis_params['nuclide_refinement'] = False
    analysis_params['spatial_refinement'] = False
    analysis_params['CFL_refinement'] = False
    analysis_params['times_refinement'] = False

    run_params = {}
    run_params['scaled_flux'] = True
    run_params['openmc_data_path'] = '/home/luke/projects/cross-section-libraries/nndc_hdf5/'
    run_params['temperature'] = '294K'
    run_params['neutron_energy'] = 0.0253
    run_params['chain_path'] = '../../data/chain_endfb71_pwr.xml'
    run_params['fissile_nuclide'] = 'U235'
    run_params['target_element'] = 'Xe'#'Nb'
    run_params['target_isobar'] = '135'#'95'
    run_params['spacenodes'] = 500
    run_params['time_mult'] = 1
    run_params['num_nuclides'] = 5
    run_params['data_gen_option'] = 'hardcoded'
    run_params['final_time'] = 5*60 #1.25*24*3600 + 100000 #56340000 #1.25*24*3600 + 100000 #  + 100000 #56340000 #1.25*24*3600 + 100_000 #29_210_400 #1.25 * 24 * 3600 #5
    run_params['solver_method'] = 'PDE'
    run_params['flux'] = 8.4e12 # 2.9e12 #1.61e13 #6e12
    run_params['frac_in'] = 0.272
    run_params['CFL_cond'] = 1
    run_params['p0'] = 7.34e6 #8e6
    run_params['power_version'] = 'constant'
    run_params['flow_version'] = 'constant'
    run_params['flux_shape'] = 'flat'
    run_params['fissile_atom_dens_cc'] = 8.41e19
    run_params['repr_loc'] = 'ex'
    run_params['reprocessing'] = {'Xe': 1/20*0,
                                  'I' : 1/20*0,
                                  'Te': 1/20*0,
                                  'Sb': 1/20*0,
                                  'Nb': 2.3e-8*100,
                                  'Tc': 1.61e-3*0,
                                  'Mo': 2.19e-7*0}

    # https://www.osti.gov/servlets/purl/1488384
    #run_params['linear_flow_rate'] = 600 # cm/s
    speed_adjustment = 10/9.98 # for flux analysis
    run_params['linear_flow_rate'] = speed_adjustment * 21.75 #21.75 # cm/s
    #run_params['net_length'] = run_params['residence_time'] * 600 #608.06 cm
    run_params['net_length'] = 608.06
    run_params['residence_time'] = run_params['net_length'] / run_params['linear_flow_rate'] # s
    #run_params['vol_flow_rate'] = 75708
    #run_params['fuel_fraction'] = 0.225
    #run_params['core_rad'] = 140.335 / 2
    run_params['net_cc_vol'] = 2_116_111
    run_params['J_per_fiss'] = 3.2e-11


    allowed_params = {}
    available_temperatures = ['294K']
    available_energies = [0.0253, 500_000, 14_000_000]
    available_flux_shapes = ['flat', 'sin']
    available_data = ['openmc', 'hardcoded']
    available_methods = ['ODE', 'PDE']
    available_versions = ['constant', 'sin', 'neg_exp', 'msre', 'step', 'sqwv']
    available_flows = ['constant', 'expdec', 'lindec', 'expinc']
    available_locs = ['in', 'ex']

    allowed_params['temperature'] = available_temperatures
    allowed_params['repr_loc'] = available_locs
    allowed_params['neutron_energy'] = available_energies
    allowed_params['data_gen_option'] = available_data
    allowed_params['solver_method'] = available_methods
    allowed_params['power_version'] = available_versions
    allowed_params['flux_shape'] = available_flux_shapes
    allowed_params['flow_version'] = available_flows

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
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[0, -1, int(run_params['spacenodes']*run_params['frac_in'])], surf_opt=plotting_params['surf_plot'])

    if analysis_params['nuclide_refinement']:
        print('-' * 50)
        nucs = [1, 5]
        max_nuc = np.max(nucs)
        if max_nuc > 5 and run_params['data_gen_option'] == 'hardcoded':
            raise UserWarning(
                'Hardcoded data only has 5 nuclides, cannot refine to more than 5')
        data_dict = analyzer.nuclide_refinement(nucs=nucs)
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])

    if analysis_params['spatial_refinement']:
        print('-' * 50)
        spatial_nodes = [50, 100, 200, 500, 1000, 2000, 4000]
        data_dict = analyzer.spatial_refinement(spatial_nodes)
        plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])

    if analysis_params['times_refinement']:
        print('-' * 50)
        time_nodes = [5, 50, 500, 5000, 50000, 500000]
        data_dict = analyzer.time_refinement(time_nodes)
        #plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])


    if analysis_params['CFL_refinement']:
        print('-' * 50)
        #CFL_nodes = [1000, 500, 250, 100, 50, 10]
        CFL_nodes = [1000, 500, 250, 100, 50, 10, 5, 1]
        #CFL_nodes = [100, 50, 25, 10, 5, 2.5, 1, 0.5, 0.25, 0.1]
        data_dict = analyzer.CFL_refinement(CFL_nodes)
        #plotter_tool.plot_gen(data_dict, spatial_eval_positions=[])

    print(f'Took {round(time() - start, 1)} seconds')
