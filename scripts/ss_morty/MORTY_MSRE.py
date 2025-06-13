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

# # Flowpath parameters
# volumetric flow rates for each flowpath [cm3/s]
nu1 = 75708.2
nu2 = 75708.2
nu3 = 75708.2
nu4 = 75708.2
nu5 = 3154.508  # volute to spray ring
nu6 = 3154.508  # spray ring to pump bowl pool
nu7 = 4100.861  # pump bowl pool to volute through "teeth" (Assume 65 gpm to maintain steady state)
nu8 = 946.353  # volute to shaft leak
nu9 = 946.353  # shaft to pump bowl pool

flow_mult = 1
# volumetric flow rates for each flowpath [cm3/s]
nu1 = 75708.2
nu2 = 75708.2 * flow_mult
nu3 = 75708.2 * flow_mult
nu4 = 75708.2 * flow_mult
nu5 = 3154.508 * flow_mult  # volute to spray ring
nu6 = 3154.508  * flow_mult # spray ring to pump bowl pool
nu7 = 4100.861  * flow_mult # pump bowl pool to volute through "teeth" (Assume 65 gpm to maintain steady state)
nu8 = 946.353  * flow_mult # volute to shaft leak
nu9 = 946.353  * flow_mult # shaft to pump bowl pool

# volume in each region [cm3]
V1 = 666011.136    # core
V2 = 381993.632     # leg A
V3 = 31148.48     # volute
V4 = 878387.136    # leg B
V5 = 5663.36     # spray ring
V6 = 82118.72     # pump bowl
V7 = 5337.22     # shaft leak

vol_mult = 1.1125774913967554
# volume in each region [cm3]
V1 = 2116111*0.272    # core
V2 = 381993.632  * vol_mult   # leg A
V3 = 31148.48     * vol_mult   # volute
V4 = 878387.136    * vol_mult   # leg B
V5 = 5663.36     * vol_mult   # spray ring
V6 = 82118.72     * vol_mult   # pump bowl
V7 = 5337.22     * vol_mult   # shaft leak

Vnet = V1 + V2 + V3 + V4 + V5 + V6 + V7  # net volume of the system
print(f'Core volume: {V1} cm3')
print(f'External volume: {V2 + V3 + V4 + V5 + V6 + V7} cm3')

print(f'Net volume: {Vnet} cm3')
print(f'In-core fraction: {V1/Vnet:.4f} of total volume')

# residence times in each region [s]
t1 = 8.79708058
t2 = 5.045604466
t3 = 0.411428088
t4 = 11.60227209
t5 = 1.795322567
t6 = 20.02475171
t7 = 5.639780432

t1 = V1/nu1
t2 = V2/nu2
t3 = V3/nu3
t4 = V4/nu4
t5 = V5/nu5
t6 = V6/nu7
t7 = V7/nu9

fvb = 0.94878119
fvsl = 0.01181973
fvsr = 0.03939908

tvtoc = (t3 + fvsr * t5 + fvsl * t7 + (1-fvb) * t6 + fvb*t4) / fvb

tbar = t1 + t2 + tvtoc
print(f'In-core residence time: {t1:.4f} s')
print(f'Average residence time in the system: {tbar} s')
print(f'Residence time fraction in the core: {t1/tbar:.4f} of total residence time')
print(f'Residence time fraction in the external system: {(t2 + tvtoc)/tbar:.4f} of total residence time')

# Pick which isobar you want to model:
decay_chain_isobar = 135

# Power of reactor and "Power constant" (PC):
P_mu = 7340000
P_sigma = 0 #P_mu / 100
PC_mu = 31000000000 # 1.63086584e+10      # <---Optimized value;   Default: 31000000000
PC_sigma = 0 #PC_mu/10

# Core neutronics for production/losses due to neutron capture
# Change to 0 if you don't want to account for losses due to transmutations in the core
phi_th = 8.4E12 # average thermal neutron flux per cm2 per s

# Add Thermal Microscopic Cross Sections [cm2] to nuclear data
ng_I135 = 80.53724E-24
ng_Xe135 = 2666886.8E-24
ng_Xe135_m1 = 0#10187238E-24
ng_Nb95 = 7.0470486E-24
ng_Nb95_m1 = 2.49332E-24

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

# Core volume [cm3]:
#V1_mu = 666011.136
V1_mu = 2116111*0.272    # core
V1_sigma = 0

# Pick Solver Method: Should be one of - 'hybr' , 'lm' , 'broyden1' , 'broyden2' , 'anderson' , 'linearmixing' ,
# 'diagbroyden' , 'excitingmixing' , 'krylov' , 'df-sane'
solver_method = 'hybr'

# Initiate chemical removal constant in region 6: spray into the pump bowl pool
r_6a = r_6b = r_6c = r_6d = 0.0
# Initiate chemical removal constant in regions 2 and 4: deposition on piping in Leg A, Leg B/HX
r_2a = r_4a = r_2b = r_4b = r_2c = r_4c = r_2d = r_4d = 0.0
# Initiate chemical removal constant in regions 1, 3, 5, and 7: core, volute, spray ring, and shaft leak
r_1a = r_3a = r_5a = r_7a = r_1b = r_3b = r_5b = r_7b = r_1c = r_3c = r_5c = r_7c = r_1d = r_3d = r_5d = r_7d = r_1d_m1 = 0.0

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


power_min, power_max = -10, 10
# V1_lower, V1_upper = 0.0, 3000000
# V1_rvs = st.truncnorm(
#    (V1_lower - V1_mu) / V1_sigma, (V1_upper - V1_mu) / V1_sigma, loc=V1_mu, scale=V1_sigma)
# V1_dist = V1_rvs.rvs(1000)
list_rho = []
list_V1 = []
list_P = []
list_PC = []
list_halogen_SP = []
list_noblegas_SP = []
list_alkali_SP = []
list_alkalineearth_SP = []
list_NbMoTcRu_SP = []
list_Sn_dep = []
list_Sb_dep = []
list_Te_dep = []
list_NbMoTcRu_dep = []
list_dpmga_6 = []
list_dpmgb_6 = []
list_dpmgc_6 = []
list_dpmgd_m1_6 = []
list_dpmgd_6 = []
list_dpmgd_m1_8 = []
list_dpmgd_8 = []
list_dpmgd_m1_7 = []
list_dpmgd_7 = []
list_dpmgd_m1_6 = []
list_dpmgd_6 = []
list_dpmgd_m1_5 = []
list_dpmgd_5 = []
list_dpmgd_m1_4 = []
list_dpmgd_4 = []
list_dpmgd_m1_3 = []
list_dpmgd_3 = []
list_dpmgd_m1_2 = []
list_dpmgd_2 = []
list_dpmgd_m1_1 = []
list_dpmgd_1 = []
list_dpmgd_m1_0 = []
list_dpmgd_0 = []
log_array = np.logspace(-10, 10, 1000, base=10)
power_linear_array = np.linspace(0, 7340000, 1000)
for i in range(MCSiter):
    rho = np.random.normal(rho_mu, rho_sigma)
    # V1 = np.random.choice(V1_dist)
    # V1 = V1_mu
    P = np.random.normal(P_mu, P_sigma)
#     P = power_linear_array[i]                                        # use to perturb power level from 0 to 7.34 MW
#     P = np.random.choice(power_linear_array)
#     P = shuffled_array[i]
    PC = np.random.normal(PC_mu, PC_sigma)
    random_power = np.random.uniform(power_min, power_max)
#     spray_ring_chemical_removal_halogen = 10 ** random_power
    spray_ring_chemical_removal_noblegas = 0 #10 ** random_power
#     spray_ring_chemical_removal_alkali = 10 ** random_power
#     spray_ring_chemical_removal_alkalineearth = 10 ** random_power
#     spray_ring_chemical_removal_NbMoTcRu = 10 ** random_power
#     spray_ring_chemical_removal_NbMoTcRu = log_array[i]              # use to perturb spray ring removal
#     deposition_chemical_removal_Sn = 10 ** random_power
#     deposition_chemical_removal_Sb = 10 ** random_power
#     deposition_chemical_removal_Te = 10 ** random_power
#     deposition_chemical_removal_NbMoTcRu = 10 ** random_power
#     deposition_chemical_removal_NbMoTcRu = log_array[i]


    list_rho.append(rho)
    # list_V1.append(V1)
    list_P.append(P)
    list_PC.append(PC)
    list_halogen_SP.append(spray_ring_chemical_removal_halogen)
    list_noblegas_SP.append(spray_ring_chemical_removal_noblegas)
    list_alkali_SP.append(spray_ring_chemical_removal_alkali)
    list_alkalineearth_SP.append(spray_ring_chemical_removal_alkalineearth)
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
      r_6a = spray_ring_chemical_removal_halogen
      r_6b = spray_ring_chemical_removal_noblegas
      r_6c = spray_ring_chemical_removal_alkali
      r_6d = spray_ring_chemical_removal_alkalineearth
    elif decay_chain_isobar == 90:
      isotopea = 'Br90'
      isotopeb = 'Kr90'
      isotopec = 'Rb90'
      isotoped = 'Sr90'
      r_6a = spray_ring_chemical_removal_halogen
      r_6b = spray_ring_chemical_removal_noblegas
      r_6c = spray_ring_chemical_removal_alkali
      r_6d = spray_ring_chemical_removal_alkalineearth
    elif decay_chain_isobar == 91:
      isotopea = 'Kr91'
      isotopeb = 'Rb91'
      isotopec = 'Sr91'
      isotoped_m1 = 'Y91_m1'
      isotoped = 'Y91'
      r_6a = spray_ring_chemical_removal_noblegas
      r_6b = spray_ring_chemical_removal_alkali
      r_6c = spray_ring_chemical_removal_alkalineearth
      r_6d = spray_ring_chemical_removal_rareearth
      br_c_d = 0.4116369
      br_dm1_d = 1.0
    elif decay_chain_isobar == 95:
      isotopea = 'Sr95'
      isotopeb = 'Y95'
      isotopec = 'Zr95'
      isotoped_m1 = 'Nb95_m1'
      isotoped = 'Nb95'
      r_6a = spray_ring_chemical_removal_alkalineearth
      r_6b = r_6c = spray_ring_chemical_removal_rareearth
      r_6d = spray_ring_chemical_removal_NbMoTcRu
      r_2d = r_4d = deposition_chemical_removal_NbMoTcRu
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
      r_6a = spray_ring_chemical_removal_alkalineearth
      r_6b = r_6c = spray_ring_chemical_removal_rareearth
      r_6d = spray_ring_chemical_removal_NbMoTcRu
      r_2d = r_4d = deposition_chemical_removal_NbMoTcRu
      br_c_d = 0.0495367
      br_dm1_d = 1
    elif decay_chain_isobar == 99:
      isotopea = 'Y99'
      isotopeb = 'Zr99'
      isotopec = 'Nb99'
      isotoped = 'Mo99'
      r_6a = r_6b = spray_ring_chemical_removal_rareearth
      r_6c = r_6d = spray_ring_chemical_removal_NbMoTcRu
      r_2c = r_4c = r_2d = r_4d = deposition_chemical_removal_NbMoTcRu
    elif decay_chain_isobar == 103:
      isotopea = 'Nb103'
      isotopeb = 'Mo103'
      isotopec = 'Tc103'
      isotoped = 'Ru103'
      r_6a = r_6b = r_6c = r_6d = spray_ring_chemical_removal_NbMoTcRu
      r_2a = r_4a = r_2b = r_4b = r_2c = r_4c = r_2d = r_4d = deposition_chemical_removal_NbMoTcRu
    elif decay_chain_isobar == 106:
      isotopea = 'Mo106'
      isotopeb = 'Tc106'
      isotopec = 'Ru106'
      isotoped = 'Rh106'
      r_6a = r_6b = r_6c = spray_ring_chemical_removal_NbMoTcRu
      r_6d = spray_ring_chemical_removal_noblemetal
      r_2a = r_4a = r_2b = r_4b = r_2c = r_4c = deposition_chemical_removal_NbMoTcRu
    elif decay_chain_isobar == 111:
      isotopea = 'Rh111'
      isotopeb = 'Pd111_m1'
      isotopec = 'Pd111'
      isotoped = 'Ag111'
      r_6a = r_6b = r_6c = r_6d = spray_ring_chemical_removal_noblemetal
      r_2a = r_4a = r_2b = r_4b = r_2c = r_4c = r_2d = r_4d = deposition_chemical_removal_noblemetal
    elif decay_chain_isobar == 129:
      isotopea = 'Sn129_m1'
      isotopeb = 'Sb129'
      isotopec = 'Te129_m1'
      isotoped = 'Te129'
      r_6a = spray_ring_chemical_removal_Sn
      r_6b = spray_ring_chemical_removal_Sb
      r_6c = r_6d = spray_ring_chemical_removal_Te
      r_2a = r_4a = deposition_chemical_removal_Sn
      r_2b = r_4b = deposition_chemical_removal_Sb
      r_2c = r_4c = r_2d = r_4d = deposition_chemical_removal_Te
    elif decay_chain_isobar == 131:
      isotopea = 'Sn131'
      isotopeb = 'Sb131'
      isotopec = 'Te131_m1'
      isotoped = 'I131'
      r_6a = spray_ring_chemical_removal_Sn
      r_6b = spray_ring_chemical_removal_Sb
      r_6c = spray_ring_chemical_removal_Te
      r_6d = spray_ring_chemical_removal_halogen
      r_2a = r_4a = deposition_chemical_removal_Sn
      r_2b = r_4b = deposition_chemical_removal_Sb
      r_2c = r_4c = deposition_chemical_removal_Te
    elif decay_chain_isobar == 132:
      isotopea = 'Sn132'
      isotopeb = 'Sb132'
      isotopec = 'Te132'
      isotoped = 'I132'
      r_6a = spray_ring_chemical_removal_Sn
      r_6b = spray_ring_chemical_removal_Sb
      r_6c = spray_ring_chemical_removal_Te
      r_6d = spray_ring_chemical_removal_halogen
      r_2a = r_4a = deposition_chemical_removal_Sn
      r_2b = r_4b = deposition_chemical_removal_Sb
      r_2c = r_4c = deposition_chemical_removal_Te
    elif decay_chain_isobar == 135:
      isotopea = 'Sb135'
      isotopeb = 'Te135'
      isotopec = 'I135'
      isotoped_m1 = 'Xe135_m1'
      isotoped = 'Xe135'
      r_6a = spray_ring_chemical_removal_Sb
      r_6b = spray_ring_chemical_removal_Te
      r_6c = spray_ring_chemical_removal_halogen
      r_6d = spray_ring_chemical_removal_noblegas
      r_2a = r_4a = deposition_chemical_removal_Sb
      r_2b = r_4b = deposition_chemical_removal_Te
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
      r_6a = spray_ring_chemical_removal_Te
      r_6b = spray_ring_chemical_removal_halogen
      r_6c = spray_ring_chemical_removal_noblegas
      r_6d = spray_ring_chemical_removal_alkali
      r_2a = r_4a = deposition_chemical_removal_Te
    elif decay_chain_isobar == 140:
      isotopea = 'Xe140'
      isotopeb = 'Cs140'
      isotopec = 'Ba140'
      isotoped = 'La140'
      r_6a = spray_ring_chemical_removal_noblegas
      r_6b = spray_ring_chemical_removal_alkali
      r_6c = spray_ring_chemical_removal_alkalineearth
      r_6d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 141:
      isotopea = 'Cs141'
      isotopeb = 'Ba141'
      isotopec = 'La141'
      isotoped = 'Ce141'
      r_6a = spray_ring_chemical_removal_alkali
      r_6b = spray_ring_chemical_removal_alkalineearth
      r_6c = r_6d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 144:
      isotopea = 'Ba144'
      isotopeb = 'La144'
      isotopec = 'Ce144'
      isotoped = 'Pr144'
      r_6a = spray_ring_chemical_removal_alkalineearth
      r_6b = r_6c = r_6d = spray_ring_chemical_removal_rareearth
    elif decay_chain_isobar == 147:
      isotopea = 'La147'
      isotopeb = 'Ce147'
      isotopec = 'Pr147'
      isotoped = 'Nd147'
      r_6a = r_6b = r_6c = r_6d = spray_ring_chemical_removal_rareearth
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

    FYa = PC * P * Ya
    FYb = PC * P * Yb
    FYc = PC * P * Yc
    FYd_m1 = PC * P * Yd_m1
    FYd = PC * P * Yd


    nui = np.array([nu1, nu2, nu3, nu4, nu5, nu6, nu7, nu8, nu9])


    Vi = np.array([V1, V2, V3, V4, V3, V5, V6, V3, V7])

    # # # Decay Chain: Isotope A
    # source term in each region [atoms/cm3/s]

    S = {'a': [FYa/V1, 0/V2, 0/V3, 0/V4, 0/V5, 0/V6, 0/V7]}

    # loss (or removal) constants for each region [s-1]: decay, chemical removal, and/or neutron absorption
    mu = {
      'a': [lama + r_1a, lama + r_2a, lama + r_3a, lama + r_4a, lama + r_5a, lama + r_6a, lama + r_7a],
      'b': [lamb + r_1b, lamb + r_2b, lamb + r_3b, lamb + r_4b, lamb + r_5b, lamb + r_6b, lamb + r_7b],
      'c': [lamc + r_1c, lamc + r_2c, lamc + r_3c, lamc + r_4c, lamc + r_5c, lamc + r_6c, lamc + r_7c],
      'd_m1': [lamd_m1 + r_1d_m1, lamd_m1 + r_2d, lamd_m1 + r_3d, lamd_m1 + r_4d, lamd_m1 + r_5d, lamd_m1 + r_6d, lamd_m1 + r_7d],
      'd': [lamd + r_1d, lamd + r_2d, lamd + r_3d, lamd + r_4d, lamd + r_5d, lamd + r_6d, lamd + r_7d],
    }

    def system(z, isotope):
      """arbitrary system of nonlinear equations"""
      N1, N2, N3, N4, N5, N6, N7, N8, N9 = z
      mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu[isotope]
      S1, S2, S3, S4, S5, S6, S7 = S[isotope]

      F = np.empty(9)
      F[0] = (S1 / mu1) + ((N4 - S1 / mu1) * exp(-mu1 * V1/nu1)) - N1
      F[1] = ((N1 - S2 / mu2) * exp(-mu2 * V2/nu2)) + (S2 / mu2) - N2
      F[2] = S3 / mu3 - (S3 / mu3 - ((N2 * nu2 + N7 * nu7)/(nu2 + nu7))) * exp(-mu3 * V3/(nu2+nu7)) - N3
      F[3] = ((N3 - S4 / mu4) * exp(-mu4 * V4/nu4)) + (S4 / mu4) - N4
      F[4] = ((N5 - S5 / mu5) * exp(-mu5 * V5/nu5)) + (S5 / mu5) - N6
      F[5] = S6 / mu6 - (S6 / mu6 - (N6 * nu6 + N9 * nu9)/(nu6 + nu9)) * exp(-mu6 * V6/nu7) - N7
      F[6] = ((N8 - S7 / mu7) * exp(-mu7 * V7/nu9)) + (S7 / mu7) - N9
      F[7] = N3 - N5
      F[8] = N3 - N8

      return F

    mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu['a']
    muia = np.array([mu1,mu2,mu3,mu4,mu1,mu5,mu6,mu1,mu7])
    NInitialA = FYa / muia

    NInitial = np.array([1.01191462e+12, 1.01087604e+12, 1.00242180e+12, 9.99448391e+11,
                        9.86231084e+9, 9.69803811e+9, 1.03686747e+10, 1.02079404e+10, 1.01888654e+10])

    Nia = opt.root(system, NInitial, 'a', method=solver_method)  # Number of atoms per cm3 at flow boundary of each region
    Cmia = Nia.x / rho  # Number of atoms per gram of salt in each region
    dpmga = Cmia * lama * 60  # Activity [dpm] per gram of salt in each region

    # # # Decay Chain: Isotope B
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['b'] = [
      Nia.x[3] * lama + FYb / V1,
      Nia.x[0] * lama,
      ((Nia.x[1] * nu2 + Nia.x[6] * nu7) / (nu2 + nu7)) * lama,
      Nia.x[2] * lama,
      Nia.x[4] * lama,
      ((Nia.x[5] * nu6 + Nia.x[8] * nu9) / (nu6 + nu9)) * lama,
      Nia.x[7] * lama,
    ]

    mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu['b']
    muib = np.array([mu1,mu2,mu3,mu4,mu1,mu5,mu6,mu1,mu7])
    NInitialBtotal = FYa * lama / (muia * muib) + FYb / muib
    NInitialB = (Vi / sum(Vi)) * NInitialBtotal

    Nib = opt.root(system, NInitial, 'b', method=solver_method)
    Cmib = Nib.x / rho
    dpmgb = Cmib * lamb * 60

    # # # Decay Chain: Isotope C
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['c'] = [
      Nib.x[3] * lamb + FYc / V1,
      Nib.x[0] * lamb,
      ((Nib.x[1] * nu2 + Nib.x[6] * nu7) / (nu2 + nu7)) * lamb,
      Nib.x[2] * lamb,
      Nib.x[4] * lamb,
      ((Nib.x[5] * nu6 + Nib.x[8] * nu9) / (nu6 + nu9)) * lamb,
      Nib.x[7] * lamb,
    ]

    mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu['c']
    muic = np.array([mu1,mu2,mu3,mu4,mu1,mu5,mu6,mu1,mu7])
    NInitialCtotal = FYa * lama * lamb / (muia * muib * muic) + FYb * lamb / (muib * muic) + FYc / muic
    NInitialC = (Vi / sum(Vi)) * NInitialCtotal

    Nic = opt.root(system, NInitial, 'c', method=solver_method)
    Cmic = Nic.x / rho
    dpmgc = Cmic * lamc * 60

    # # # Decay Chain: Isotope D_m1
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['d_m1'] =  [FYd_m1/V1, 0/V2, 0/V3, 0/V4, 0/V5, 0/V6, 0/V7]

    mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu['d_m1']
    muid_m1 = np.array([mu1,mu2,mu3,mu4,mu1,mu5,mu6,mu1,mu7])
    NInitialD_m1 = FYd_m1 / muid_m1

    Nid_m1 = opt.root(system, NInitial, 'd_m1', method=solver_method)
    Cmid_m1 = Nid_m1.x / rho
    dpmgd_m1 = Cmid_m1 * lamd_m1 * 60

    # # # Decay Chain: Isotope D
    # # Region parameters
    # source term in each region [atoms/cm3/s]
    S['d'] = [
      br_c_d * Nic.x[3] * lamc + FYd / V1 + br_dm1_d * Nid_m1.x[3] * lamd_m1,
      br_c_d * Nic.x[0] * lamc + br_dm1_d * Nid_m1.x[0] * lamd_m1,
      ((br_c_d * Nic.x[1] * nu2 + br_c_d * Nic.x[6] * nu7) / (nu2 + nu7)) * lamc + ((br_dm1_d * Nid_m1.x[1] * nu2 + br_dm1_d * Nid_m1.x[6] * nu7) / (nu2 + nu7)) * lamd_m1,
      br_c_d * Nic.x[2] * lamc + br_dm1_d * Nid_m1.x[2] * lamd_m1,
      br_c_d * Nic.x[4] * lamc + br_dm1_d * Nid_m1.x[4] * lamd_m1,
      ((br_c_d * Nic.x[5] * nu6 + br_c_d * Nic.x[8] * nu9) / (nu6 + nu9)) * lamc + ((br_dm1_d * Nid_m1.x[5] * nu6 + br_dm1_d * Nid_m1.x[8] * nu9) / (nu6 + nu9)) * lamd_m1,
      br_c_d * Nic.x[7] * lamc + br_dm1_d * Nid_m1.x[7] * lamd_m1,
    ]

    mu1, mu2, mu3, mu4, mu5, mu6, mu7 = mu['d']
    muid = np.array([mu1,mu2,mu3,mu4,mu1,mu5,mu6,mu1,mu7])
    NInitialDtotal = FYa * lama * lamb * lamc / (muia * muib * muic * muid) + FYb * lamb * lamc / (muib * muic * muid) \
                 + FYc * lamc / (muic * muid) + FYd / muid + FYd_m1 * lamd_m1 / (muid_m1 * muid)
    NInitialD = (Vi / sum(Vi)) * NInitialDtotal

    Nid = opt.root(system, NInitial, 'd', method=solver_method)
    Cmid = Nid.x / rho
    dpmgd = Cmid * lamd * 60

    list_dpmga_6.append(dpmga[6])
    list_dpmgb_6.append(dpmgb[6])
    list_dpmgc_6.append(dpmgc[6])
    list_dpmgd_m1_6.append(dpmgd_m1[6])
    list_dpmgd_6.append(dpmgd[6])
    list_dpmgd_m1_8.append(dpmgd_m1[8])
    list_dpmgd_8.append(dpmgd[8])
    list_dpmgd_m1_7.append(dpmgd_m1[7])
    list_dpmgd_7.append(dpmgd[7])
    list_dpmgd_m1_5.append(dpmgd_m1[5])
    list_dpmgd_5.append(dpmgd[5])
    list_dpmgd_m1_4.append(dpmgd_m1[4])
    list_dpmgd_4.append(dpmgd[4])
    list_dpmgd_m1_3.append(dpmgd_m1[3])
    list_dpmgd_3.append(dpmgd[3])
    list_dpmgd_m1_2.append(dpmgd_m1[2])
    list_dpmgd_2.append(dpmgd[2])
    list_dpmgd_m1_1.append(dpmgd_m1[1])
    list_dpmgd_1.append(dpmgd[1])
    list_dpmgd_m1_0.append(dpmgd_m1[0])
    list_dpmgd_0.append(dpmgd[0])
    

# Half-lives of fission products to print
t12a = np.log(2)/lama
t12b = np.log(2)/lamb
t12c = np.log(2)/lamc
t12d_m1 = np.log(2)/lamd_m1
t12d = np.log(2)/lamd

Vi_mu = np.array([V1_mu, V2, V3, V4, V3, V5, V6, V3, V7])
total_salt_wt = sum(Vi_mu) * rho_mu / 1000

FYa_mu = PC_mu * P_mu * Ya
FYb_mu = PC_mu * P_mu * Yb
FYc_mu = PC_mu * P_mu * Yc
FYd_m1_mu = PC_mu * P_mu * Yd_m1
FYd_mu = PC_mu * P_mu * Yd

anal_mud = lamd + r_1d
anal_mud_m1 = lamd_m1 + r_1d_m1

# Direct production: Analytical solution for steady-state FP production due to fissions but not production from parent decays
anal_dpmga = FYa_mu * lama * 60 / (lama * (1000 * total_salt_wt))
anal_dpmgb = FYb_mu * lamb * 60 / (lamb * (1000 * total_salt_wt))
anal_dpmgc = FYc_mu * lamc * 60 / (lamc * (1000 * total_salt_wt))
anal_dpmgd_m1 = FYd_m1_mu * lamd_m1 * 60 / (anal_mud_m1 * (1000 * total_salt_wt))
anal_dpmgd = FYd_mu * lamd * 60 / (anal_mud * (1000 * total_salt_wt))
anal_dpmg = np.array([anal_dpmga, anal_dpmgb, anal_dpmgc, anal_dpmgd_m1, anal_dpmgd])

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