
from scipy import optimize as opt
import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
import scipy.stats as st
import openmc.deplete
import pandas as pd

# #
# # USER INPUTS
# #

# Pick which isobar you want to model:
decay_chain_isobar = 135
reactor_name = "MSRE"


# # Reactor parameters: Flowpath, region, power
if reactor_name == "MSRE":
    # volumetric flow rates for each flowpath [cm3/s]
    nu1 = 66666.6667      # excore to core
    nu2 = 66666.6667      # core to excore
    # residence times in each region [s]
    t1 = 24#9.9              # core
    t2 = 6#20.1             # excore
    print('Change residence times')
    tnet = 608.06/21.75
    t1 = tnet * 0.272
    t2 = tnet * (1-0.272)
    # volume in each region [cm3]
    V1 = 1.6e6#660000           # core
    V2 = 0.4e6#1340000          # excore
    print('Change volumes')
    Vnet = 2116111
    V1 = Vnet * 0.272
    V2 = Vnet * (1-0.272)
    rho_mu = 2.32556         # density of fuel salt [g/cm3]
    rho_sigma = 0.0
    # power [W] and flux [n/cm2-s]
    P_mu = 7340000        # 8 MWth
    P_sigma = 0           # if you want to do Monte Carlo on power level
    phi_th = 8.4E12
elif reactor_name == "MSDR":
    # volumetric flow rates for each flowpath [cm3/s]
    nu1 = 2.0655E+06      # excore to core
    nu2 = 2.0655E+06      # core to excore
    # residence times in each region [s]
    t1 = 14               # core
    t2 = 6                # excore
    # volume in each region [cm3]
    V1 = 2.89170E+07      # core
    V2 = 1.23930E+07      # excore
    rho_mu = 4.71         # density of fuel salt [g/cm3]
    rho_sigma = 0.0
    # power [W] and flux [n/cm2-s]
    P_mu = 750000000      # 750 MWth
    P_sigma = 0           # if you want to do Monte Carlo on power level
    phi_th = 1.8E13 
else:
  print('Chosen reactor is not available!') 

nui = np.array([nu1, nu2])
Vi = np.array([V1, V2])

# "Power constant" (PC):
PC_mu = 31000000000       # Default: 31000000000
PC_sigma = 0              # if you want to do Monte Carlo on power level constant

# Add Thermal Microscopic Cross Sections [cm2] to nuclear data
ng_Nb95 = 7.0470486E-24
ng_Nb95_m1 = 2.49332E-24
ng_I131 = 80.536095E-24
ng_I135 = 80.53724E-24
ng_Xe135 = 1.50E-18
ng_Xe135_m1 = 0 #10187238E-24
ng_Ce141 = 29.19615E-24
ng_Nd147 = 443.4743E-24
ng_Eu155 = 3783.1646E-24

# Number of histories for the Monte Carlo Sampling of each isobar:
MCSiter = 1

# fission products generated per second, FY [atoms/s], as a result of fissions in the core at power level, P [W]
frac233 = frac235 = frac239 = 0
# frac233 = 0.935
# frac233 = 1
# frac235 = 0.022
frac235 = 1
# frac239 = 0.043
# frac239 = 1

# Spray Ring Chemical Removal Rates
spray_ring_chemical_removal_NbMoTcRu = 0
spray_ring_chemical_removal_noblemetal = 0
spray_ring_chemical_removal_Sn = 0
spray_ring_chemical_removal_Sb = 0
spray_ring_chemical_removal_Te = 0
spray_ring_chemical_removal_halogen = 0
spray_ring_chemical_removal_noblegas = 0 # 0.0041 / 0.08212
spray_ring_chemical_removal_alkali = 0
spray_ring_chemical_removal_alkalineearth = 0
spray_ring_chemical_removal_rareearth = 0

# Alloy Deposition Chemical Removal Rates
deposition_chemical_removal_NbMoTcRu = 0
deposition_chemical_removal_noblemetal = 0
deposition_chemical_removal_Sn = 0
deposition_chemical_removal_Sb = 0
deposition_chemical_removal_Te = 0

# Density of fuel salt [g/cm3]
rho_mu = 2.32556
rho_sigma = 0.0

# Pick Solver Method: Should be one of - 'hybr' , 'lm' , 'broyden1' , 'broyden2' , 'anderson' , 'linearmixing' ,
# 'diagbroyden' , 'excitingmixing' , 'krylov' , 'df-sane'
solver_method = 'hybr'

# Initiate chemical removal constant in region 2: spray into the pump bowl pool, deposition on piping in Leg A, Leg B/HX
r_2a = r_2b = r_2c = r_2d = 0.0
# Initiate chemical removal constant in regions 1: core
r_1a = r_1b = r_1c = r_1d = 0.0

# Initiate Decay Branching Ratios
br_c_d = 1
br_dm1_d = 1


# #
# # LOAD NUCLEAR DATA FROM OPENMC
# #


# Read decay constants from OpenMC depletion chain
chain = openmc.deplete.Chain.from_xml('../../data/chain_endfb71_pwr.xml')
lam = {}
for nuclide in chain.nuclides:
   if nuclide.half_life is not None:
      lam[nuclide.name] = log(2.0) / nuclide.half_life

# Read fission product yields from OpenMC database
Y_cumulative = {}
Y_independent = {}
u233 = openmc.data.FissionProductYields('nfy-092_U_233.endf')
u235 = openmc.data.FissionProductYields('nfy-092_U_235.endf')
pu239 = openmc.data.FissionProductYields('nfy-094_Pu_239.endf')

for nuclide in u233.cumulative[0]:
   # Skip nuclides that don't appear for all sets of FPYs
   if nuclide not in u235.cumulative[0] or nuclide not in pu239.cumulative[0]:
      continue

   Y_cumulative[nuclide] = (
      u233.cumulative[0][nuclide].nominal_value * frac233 +
      u235.cumulative[0][nuclide].nominal_value * frac235 +
      pu239.cumulative[0][nuclide].nominal_value * frac239
   )
   Y_independent[nuclide] = (
      u233.independent[0][nuclide].nominal_value * frac233 +
      u235.independent[0][nuclide].nominal_value * frac235 +
      pu239.independent[0][nuclide].nominal_value * frac239
   )



# #
# # MONTE CARLO SAMPLING
# #


#power_min, power_max = -10, 10
# V1_lower, V1_upper = 0.0, 3000000
# V1_rvs = st.truncnorm(
#    (V1_lower - V1_mu) / V1_sigma, (V1_upper - V1_mu) / V1_sigma, loc=V1_mu, scale=V1_sigma)
# V1_dist = V1_rvs.rvs(1000)
list_rho = []
list_P = []
list_PC = []
list_noblegas_SP = []
list_NbMoTcRu_SP = []
list_Sn_dep = []
list_Sb_dep = []
list_Te_dep = []
list_NbMoTcRu_dep = []
list_dpmga_1 = []
list_dpmgb_1 = []
list_dpmgc_1 = []
list_dpmgd_m1_1 = []
list_dpmgd_1 = []
list_dpmga_0 = []
list_dpmgb_0 = []
list_dpmgc_0 = []
list_dpmgd_m1_0 = []
list_dpmgd_0 = []
list_atcca_1 = []
list_atccb_1 = []
list_atccc_1 = []
list_atccd_m1_1 = []
list_atccd_1 = []
list_atcca_0 = []
list_atccb_0 = []
list_atccc_0 = []
list_atccd_m1_0 = []
list_atccd_0 = []
log_array = np.logspace(-10, 10, 1000, base=10)
power_linear_array = np.linspace(0, P_mu, 1000)
for i in range(MCSiter):
    rho = np.random.normal(rho_mu, rho_sigma)
    P = np.random.normal(P_mu, P_sigma)
#     P = power_linear_array[i]                                        # use to perturb power level from 0 to 8 MW
#     P = np.random.choice(power_linear_array)
#     P = shuffled_array[i]
    PC = np.random.normal(PC_mu, PC_sigma)
    #random_power = np.random.uniform(power_min, power_max)
    spray_ring_chemical_removal_noblegas = 0# 10 ** random_power
#     spray_ring_chemical_removal_NbMoTcRu = 10 ** random_power
#     spray_ring_chemical_removal_NbMoTcRu = log_array[i]              # use to perturb spray ring removal
    # deposition_chemical_removal_Sn = 10 ** random_power
    # deposition_chemical_removal_Sb = 10 ** random_power
    # deposition_chemical_removal_Te = 10 ** random_power
#     deposition_chemical_removal_NbMoTcRu = 10 ** random_power
#     deposition_chemical_removal_NbMoTcRu = log_array[i]


    list_rho.append(rho)
    list_P.append(P)
    list_PC.append(PC)
    list_noblegas_SP.append(spray_ring_chemical_removal_noblegas)
    list_NbMoTcRu_SP.append(spray_ring_chemical_removal_NbMoTcRu)
    list_Sn_dep.append(deposition_chemical_removal_Sn)
    list_Sb_dep.append(deposition_chemical_removal_Sb)
    list_Te_dep.append(deposition_chemical_removal_Te)
    list_NbMoTcRu_dep.append(deposition_chemical_removal_NbMoTcRu)


    # TODO: Account for metastable/ground yields when they are being grouped
    # together. Right now they are being added manually

    # #
    # # LOAD VARIABLES BASED ON CHOSEN ISOBAR
    # #


    if decay_chain_isobar == 89:
      isotopea = 'Br89'
      isotopeb = 'Kr89'
      isotopec = 'Rb89'
      isotoped = 'Sr89'
      r_2a = spray_ring_chemical_removal_halogen
      r_2b = spray_ring_chemical_removal_noblegas
      r_2c = spray_ring_chemical_removal_alkali
      r_2d = spray_ring_chemical_removal_alkalineearth
    elif decay_chain_isobar == 90:
      isotopea = 'Br90'
      isotopeb = 'Kr90'
      isotopec = 'Rb90'
      isotoped = 'Sr90'
      r_2a = spray_ring_chemical_removal_halogen
      r_2b = spray_ring_chemical_removal_noblegas
      r_2c = spray_ring_chemical_removal_alkali
      r_2d = spray_ring_chemical_removal_alkalineearth
    elif decay_chain_isobar == 91:
      isotopea = 'Kr91'
      isotopeb = 'Rb91'
      isotopec = 'Sr91'
      isotoped = 'Y91'
      r_2a = spray_ring_chemical_removal_noblegas
      r_2b = spray_ring_chemical_removal_alkali
      r_2c = spray_ring_chemical_removal_alkalineearth
      r_2d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 95:
      isotopea = 'Sr95'
      isotopeb = 'Y95'
      isotopec = 'Zr95'
      isotoped_m1 = 'Nb95_m1'
      isotoped = 'Nb95'
      r_2a = spray_ring_chemical_removal_alkalineearth
      r_2b = spray_ring_chemical_removal_rareearth
      r_2d = spray_ring_chemical_removal_NbMoTcRu
      r_2d = deposition_chemical_removal_NbMoTcRu
      r_1d = ng_Nb95
      r_1d_m1 = ng_Nb95_m1
      br_c_d = 0.9891978
      br_dm1_d = 0.944
    elif decay_chain_isobar == 97:
      isotopea = 'Sr97'
      isotopeb = 'Y97'
      isotopec = 'Zr97'
      isotoped_m1 = 'Nb97_m1'
      isotoped = 'Nb97'
      r_2a = spray_ring_chemical_removal_alkalineearth
      r_2b = spray_ring_chemical_removal_rareearth
      r_2d = spray_ring_chemical_removal_NbMoTcRu
      r_2d = deposition_chemical_removal_NbMoTcRu
      br_c_d = 0.0495367
      br_dm1_d = 1
    elif decay_chain_isobar == 99:
      isotopea = 'Y99'
      isotopeb = 'Zr99'
      isotopec = 'Nb99'
      isotoped = 'Mo99'
      r_2a = r_2b = spray_ring_chemical_removal_rareearth
      r_2c = r_2d = spray_ring_chemical_removal_NbMoTcRu + deposition_chemical_removal_NbMoTcRu
    elif decay_chain_isobar == 103:
      isotopea = 'Nb103'
      isotopeb = 'Mo103'
      isotopec = 'Tc103'
      isotoped = 'Ru103'
      r_2a = r_2b = r_2c = r_2d = spray_ring_chemical_removal_NbMoTcRu + deposition_chemical_removal_NbMoTcRu
    elif decay_chain_isobar == 106:
      isotopea = 'Mo106'
      isotopeb = 'Tc106'
      isotopec = 'Ru106'
      isotoped = 'Rh106'
      r_2a = r_2b = r_2c = spray_ring_chemical_removal_NbMoTcRu + deposition_chemical_removal_NbMoTcRu
      r_2d = spray_ring_chemical_removal_noblemetal
    elif decay_chain_isobar == 111:
      isotopea = 'Rh111'
      isotopeb = 'Pd111_m1'
      isotopec = 'Pd111'
      isotoped = 'Ag111'
      r_2a = r_2b = r_2c = r_2d = spray_ring_chemical_removal_noblemetal + deposition_chemical_removal_noblemetal
    elif decay_chain_isobar == 114:
      isotopea = 'Ru114'
      isotopeb = 'Rh114'
      isotopec = 'Pd114'
      isotoped = 'Ag114'
      r_2a = r_2b = r_2c = r_2d = spray_ring_chemical_removal_noblemetal + deposition_chemical_removal_noblemetal
    elif decay_chain_isobar == 129:
      isotopea = 'Sn129_m1'
      isotopeb = 'Sb129'
      isotopec = 'Te129_m1'
      isotoped = 'Te129'
      r_2a = spray_ring_chemical_removal_Sn + deposition_chemical_removal_Sn
      r_2b = spray_ring_chemical_removal_Sb + deposition_chemical_removal_Sb
      r_2c = r_2d = spray_ring_chemical_removal_Te + deposition_chemical_removal_Te
    elif decay_chain_isobar == 131:
      isotopea = 'Sn131'
      isotopeb = 'Sb131'
      isotopec = 'Te131_m1'
      isotoped = 'I131'
      r_2a = spray_ring_chemical_removal_Sn + deposition_chemical_removal_Sn
      r_2b = spray_ring_chemical_removal_Sb + deposition_chemical_removal_Sb
      r_2c = spray_ring_chemical_removal_Te + deposition_chemical_removal_Te
      r_2d = spray_ring_chemical_removal_halogen
      r_1d = phi_th * ng_I131  
    elif decay_chain_isobar == 132:
      isotopea = 'Sn132'
      isotopeb = 'Sb132'
      isotopec = 'Te132'
      isotoped = 'I132'
      r_2a = spray_ring_chemical_removal_Sn + deposition_chemical_removal_Sn
      r_2b = spray_ring_chemical_removal_Sb + deposition_chemical_removal_Sb
      r_2c = spray_ring_chemical_removal_Te + deposition_chemical_removal_Te
      r_2d = spray_ring_chemical_removal_halogen
    elif decay_chain_isobar == 135:
      isotopea = 'Sb135'
      isotopeb = 'Te135'
      isotopec = 'I135'
      isotoped_m1 = 'Xe135_m1'
      isotoped = 'Xe135'
      r_2a = spray_ring_chemical_removal_Sb + deposition_chemical_removal_Sb
      r_2b = spray_ring_chemical_removal_Te + deposition_chemical_removal_Te
      r_2c = spray_ring_chemical_removal_halogen
      r_2d = spray_ring_chemical_removal_noblegas
      r_1c = phi_th * ng_I135
      r_1d = phi_th * ng_Xe135
      r_1d_m1 = phi_th * ng_Xe135_m1
      br_c_d = 0.8349109
      br_dm1_d = 0.997
    elif decay_chain_isobar == 137:
      isotopea = 'Te137'
      isotopeb = 'I137'
      isotopec = 'Xe137'
      isotoped = 'Cs137'
      r_2a = spray_ring_chemical_removal_Te + deposition_chemical_removal_Te
      r_2b = spray_ring_chemical_removal_halogen
      r_2c = spray_ring_chemical_removal_noblegas
      r_2d = spray_ring_chemical_removal_alkali
    elif decay_chain_isobar == 140:
      isotopea = 'Xe140'
      isotopeb = 'Cs140'
      isotopec = 'Ba140'
      isotoped = 'La140'
      r_2a = spray_ring_chemical_removal_noblegas
      r_2b = spray_ring_chemical_removal_alkali
      r_2c = spray_ring_chemical_removal_alkalineearth
      r_2d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 141:
      isotopea = 'Cs141'
      isotopeb = 'Ba141'
      isotopec = 'La141'
      isotoped = 'Ce141'
      r_2a = spray_ring_chemical_removal_alkali
      r_2b = spray_ring_chemical_removal_alkalineearth
      r_2c = r_6d = spray_ring_chemical_removal_rareearth
      r_1d = phi_th * ng_Ce141
    elif decay_chain_isobar == 144:
      isotopea = 'Ba144'
      isotopeb = 'La144'
      isotopec = 'Ce144'
      isotoped = 'Pr144'
      r_2a = spray_ring_chemical_removal_alkalineearth
      r_2b = r_2c = r_2d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 147:
      isotopea = 'La147'
      isotopeb = 'Ce147'
      isotopec = 'Pr147'
      isotoped = 'Nd147'
      r_2a = r_2b = r_2c = r_2d = spray_ring_chemical_removal_rareearth
      r_1d = phi_th * ng_Nd147 
    elif decay_chain_isobar == 155:
      isotopea = 'Nd155'
      isotopeb = 'Pm155'
      isotopec = 'Sm155'
      isotoped = 'Eu155'
      r_2a = r_2b = r_2c = r_2d = spray_ring_chemical_removal_rareearth
      r_1d = phi_th * ng_Eu155
    else:
      print('Chosen decay chain isobar is not available!')    

    # # Decay Chain Parameters
    # decay constant for each nuclide in the decay chain [s-1]

    lama = lam[isotopea]
    lamb = lam[isotopeb]
    lamc = lam[isotopec]
    lamd_m1 = lam[isotoped_m1]
    lamd = lam[isotoped]

    Ya = Y_cumulative[isotopea]
    Yb = Y_independent[isotopeb]
    Yc = Y_independent[isotopec]
    Yd_m1 = Y_cumulative[isotoped_m1] # I use indepdendent here
    print('Using independent yield for Yd_m1')
    Yd_m1 = Y_independent[isotoped_m1]
    Yd = Y_independent[isotoped]
    
    FYa = PC * P * Ya   # atoms/s = fissions/J * J/s * yield_fraction
    FYb = PC * P * Yb
    FYc = PC * P * Yc
    FYd_m1 = PC * P * Yd_m1
    FYd = PC * P * Yd

    # # # Decay Chain: Isotope A
    # source term in each region [atoms/cm3/s]

    S = {'a': [FYa/V1, 0/V2]}

    # loss (or removal) constants for each region [s-1]: decay, chemical removal, and/or neutron absorption
    mu = {
      'a': [lama + r_1a, lama + r_2a],
      'b': [lamb + r_1b, lamb + r_2b],
      'c': [lamc + r_1c, lamc + r_2c],
      'd_m1': [lamd_m1 + r_1d, lamd_m1 + r_2d],
      'd': [lamd + r_1d, lamd + r_2d],
    }

    def system(z, isotope):
      """arbitrary system of nonlinear equations"""
      N1, N2 = z
      mu1, mu2 = mu[isotope]
      S1, S2 = S[isotope]

      F = np.empty(2)
      F[0] = (S1 / mu1) + ((N2 - S1 / mu1) * exp(-mu1 * V1/nu1)) - N1
      F[1] = (S2 / mu2) + ((N1 - S2 / mu2) * exp(-mu2 * V2/nu2)) - N2

      return F

    mu1, mu2 = mu['a']
    muia = np.array([mu1,mu2])
    NInitialA = FYa / muia

    NInitial = np.array([1.01191462e+12, 1.01087604e+12])

    Nia = opt.root(system, NInitial, 'a', method=solver_method)  # Number of atoms per cm3 at flow boundary of each region
    atcca = Nia.x
    Cmia = Nia.x / rho  # Number of atoms per gram of salt in each region
    dpmga = Cmia * lama * 60  # Activity [dpm] per gram of salt in each region

    # # # Decay Chain: Isotope B
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['b'] = [
      Nia.x[1] * lama + FYb / V1,
      Nia.x[0] * lama,
    ]

    mu1, mu2 = mu['b']
    muib = np.array([mu1,mu2])

    Nib = opt.root(system, NInitial, 'b', method=solver_method)
    atccb = Nib.x
    Cmib = Nib.x / rho
    dpmgb = Cmib * lamb * 60

    # # # Decay Chain: Isotope C
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['c'] = [
      Nib.x[1] * lamb + FYc / V1,
      Nib.x[0] * lamb,
    ]

    mu1, mu2 = mu['c']
    muic = np.array([mu1,mu2])

    Nic = opt.root(system, NInitial, 'c', method=solver_method)
    atccc = Nic.x
    Cmic = Nic.x / rho
    dpmgc = Cmic * lamc * 60

    # # # Decay Chain: Isotope D_m1
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['d_m1'] =  [FYd_m1/V1, 0/V2]

    mu1, mu2 = mu['d_m1']
    muid_m1 = np.array([mu1,mu2])

    Nid_m1 = opt.root(system, NInitial, 'd_m1', method=solver_method)
    atccd_m1 = Nid_m1.x
    Cmid_m1 = Nid_m1.x / rho
    dpmgd_m1 = Cmid_m1 * lamd_m1 * 60

    # # # Decay Chain: Isotope D
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['d'] = [
      br_c_d * Nic.x[1] * lamc + FYd / V1 + br_dm1_d * Nid_m1.x[1] * lamd_m1,
      br_c_d * Nic.x[0] * lamc + br_dm1_d * Nid_m1.x[0] * lamd_m1,
      ]

    mu1, mu2 = mu['d']
    muid = np.array([mu1,mu2])

    Nid = opt.root(system, NInitial, 'd', method=solver_method)
    atccd = Nid.x
    Cmid = Nid.x / rho
    dpmgd = Cmid * lamd * 60
    
    list_dpmga_1.append(dpmga[1])
    list_dpmgb_1.append(dpmgb[1])
    list_dpmgc_1.append(dpmgc[1])
    list_dpmgd_m1_1.append(dpmgd_m1[1])
    list_dpmgd_1.append(dpmgd[1])

    list_dpmga_0.append(dpmga[0])
    list_dpmgb_0.append(dpmgb[0])
    list_dpmgc_0.append(dpmgc[0])
    list_dpmgd_m1_0.append(dpmgd_m1[0])
    list_dpmgd_0.append(dpmgd[0])
    
    list_atcca_1.append(atcca[1])
    list_atccb_1.append(atccb[1])
    list_atccc_1.append(atccc[1])
    list_atccd_m1_1.append(atccd_m1[1])
    list_atccd_1.append(atccd[1])

    list_atcca_0.append(atcca[0])
    list_atccb_0.append(atccb[0])
    list_atccc_0.append(atccc[0])
    list_atccd_m1_0.append(atccd_m1[0])
    list_atccd_0.append(atccd[0])
    

# Half-lives of fission products to print
t12a = np.log(2)/lama
t12b = np.log(2)/lamb
t12c = np.log(2)/lamc
t12d_m1 = np.log(2)/lamd_m1
t12d = np.log(2)/lamd

Vi_mu = np.array([V1, V2])
total_salt_wt = sum(Vi_mu) * rho_mu / 1000   # kg

FYa_mu = PC_mu * P_mu * Ya     # atoms/s
FYb_mu = PC_mu * P_mu * Yb
FYc_mu = PC_mu * P_mu * Yc
FYd_m1_mu = PC_mu * P_mu * Yd_m1
FYd_mu = PC_mu * P_mu * Yd

anal_mud = lamd + r_1d
anal_mud_m1 = lamd_m1 + r_1d

# Direct production: Analytical solution for steady-state FP production due to fissions but not production from parent decays
anal_dpmga = FYa_mu * lama * 60 / (lama * (1000 * total_salt_wt))
anal_dpmgb = FYb_mu * lamb * 60 / (lamb * (1000 * total_salt_wt))
anal_dpmgc = FYc_mu * lamc * 60 / (lamc * (1000 * total_salt_wt))
anal_dpmgd_m1 = FYd_m1_mu * lamd_m1 * 60 / (anal_mud_m1 * (1000 * total_salt_wt))
anal_dpmgd = FYd_mu * lamd * 60 / (anal_mud * (1000 * total_salt_wt))
anal_dpmg = np.array([anal_dpmga, anal_dpmgb, anal_dpmgc, anal_dpmgd_m1, anal_dpmgd])

# Direct production: Analytical solution for steady-state FP production due to fissions but not production from parent decays
anal_atcca = FYa_mu / (lama * sum(Vi_mu))
anal_atccb = FYb_mu / (lamb * sum(Vi_mu))
anal_atccc = FYc_mu / (lamc * sum(Vi_mu))
anal_atccd_m1 = FYd_m1_mu / (lamd_m1 * sum(Vi_mu))
anal_atccd = FYd_mu / (lamd * sum(Vi_mu))
anal_atcc = np.array([anal_atcca, anal_atccb, anal_atccc, anal_atccd_m1, anal_atccd])

# Cumulative production: Equation 2.114 (in 1981 Benedict - Nuclear Chemical Engineering, Ch 2) for steady-state production
eqn2114_a = FYa_mu/(lama)
eqn2114_b = FYa_mu * lama/(lama*lamb) + FYb_mu/lamb
eqn2114_c = FYa_mu * lama*lamb/(lama*lamb*lamc) + FYb_mu * lamb/(lamb*lamc) + FYc_mu/(lamc)
eqn2114_d_m1 = FYd_m1_mu /(anal_mud_m1)
eqn2114_d = FYa_mu * lama*lamb*lamc*br_c_d/(lama*lamb*lamc*(anal_mud)) + FYb_mu * lamb*lamc*br_c_d/(lamb*lamc*(anal_mud)) + FYc_mu * lamc*br_c_d/(lamc*(anal_mud)) + FYd_m1_mu * lamd_m1*br_dm1_d/((anal_mud)*(anal_mud_m1)) + FYd_mu/(anal_mud)
eqn2114_dpmga = eqn2114_a * lama * 60 / (1000 * total_salt_wt)
eqn2114_dpmgb = eqn2114_b * lamb * 60 / (1000 * total_salt_wt)
eqn2114_dpmgc = eqn2114_c * lamc * 60 / (1000 * total_salt_wt)
eqn2114_dpmgd_m1 = eqn2114_d_m1 * lamd_m1 * 60 / (1000 * total_salt_wt)
eqn2114_dpmgd = eqn2114_d * lamd * 60 / (1000 * total_salt_wt)
cum_anal_dpmg = np.array([eqn2114_dpmga, eqn2114_dpmgb, eqn2114_dpmgc, eqn2114_dpmgd_m1, eqn2114_dpmgd])

eqn2114_atcca = eqn2114_a / sum(Vi_mu)
eqn2114_atccb = eqn2114_b / sum(Vi_mu)
eqn2114_atccc = eqn2114_c / sum(Vi_mu)
eqn2114_atccd_m1 = eqn2114_d_m1 / sum(Vi_mu)
eqn2114_atccd = eqn2114_d / sum(Vi_mu)
cum_anal_atcc = np.array([eqn2114_atcca, eqn2114_atccb, eqn2114_atccc, eqn2114_atccd_m1, eqn2114_atccd])

print('Minimum Concentration of ', isotopea, ':', "{:.3E}".format(np.min(Nia.x)), ' atoms/cm3')
print('Maximum Concentration of ', isotopea, ':', "{:.3E}".format(np.max(Nia.x)), ' atoms/cm3')
print('Average Concentration of ', isotopea, ':', "{:.3E}".format(np.mean(Nia.x)), ' atoms/cm3')
print('Minimum Concentration of ', isotopeb, ':', "{:.3E}".format(np.min(Nib.x)), ' atoms/cm3')
print('Maximum Concentration of ', isotopeb, ':', "{:.3E}".format(np.max(Nib.x)), ' atoms/cm3')
print('Average Concentration of ', isotopeb, ':', "{:.3E}".format(np.mean(Nib.x)), ' atoms/cm3')
print('Minimum Concentration of ', isotopec, ':', "{:.3E}".format(np.min(Nic.x)), ' atoms/cm3')
print('Maximum Concentration of ', isotopec, ':', "{:.3E}".format(np.max(Nic.x)), ' atoms/cm3')
print('Average Concentration of ', isotopec, ':', "{:.3E}".format(np.mean(Nic.x)), ' atoms/cm3')
print('Minimum Concentration of ', isotoped_m1, ':', "{:.3E}".format(np.min(Nid_m1.x)), ' atoms/cm3')
print('Maximum Concentration of ', isotoped_m1, ':', "{:.3E}".format(np.max(Nid_m1.x)), ' atoms/cm3')
print('Average Concentration of ', isotoped_m1, ':', "{:.3E}".format(np.mean(Nid_m1.x)), ' atoms/cm3')
print('Minimum Concentration of ', isotoped, ':', "{:.3E}".format(np.min(Nid.x)), ' atoms/cm3')
print('Maximum Concentration of ', isotoped, ':', "{:.3E}".format(np.max(Nid.x)), ' atoms/cm3')
print('Average Concentration of ', isotoped, ':', "{:.3E}".format(np.mean(Nid.x)), ' atoms/cm3')
print('-'*50)