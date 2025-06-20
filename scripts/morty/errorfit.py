import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 5
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.grid.which"] = "major"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 6.0
plt.rcParams["ytick.major.size"] = 6.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["figure.autolayout"] = True
plt.rcParams['savefig.dpi'] = 600

ss_log = False
peak_log = False

chem = True
flow_scaled = False
flow_unscaled = False

separate_marker_plot = False

length = 608.06

if chem:
    xs = [0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
    peak = [7.46, 6.0, 4.52, 3.03, 1.520, 0.003]
    ss = [0.882, 0.577, 0.33, 0.155, 0.04375, 1.17e-6]
elif flow_scaled:
    #xs = [21.7e1, 21.7, 21.7e-1, 21.7e-2, 21.7e-3, 21.7*5e-4, 21.75*2.5e-4, 21.7e-4, 21.7*5e-5, 21.7e-5]
    #peak = [0.00036, 0.003, 0.027, 0.3, 2.3, 5.08, 7.64, 15.5, 23.4, 49]
    #ss = [1.6e-7, 7.5e-6, 0.002, 0.06, 0.78, 2.43, 6.0, 0.43, 15.56, 49] # 0.43, 15, 49 negative
    xs = [21.7e1, 21.7, 21.7e-1, 21.7e-2, 21.7e-3, 21.7*5e-4, 21.75*2.5e-4, 21.7*5e-5, 21.7e-5]
    peak = [0.00036, 0.003, 0.027, 0.3, 2.3, 5.08, 7.64, 23.4, 49]
    ss = [1.6e-7, 7.5e-6, 0.002, 0.06, 0.78, 2.43, 6.0, 15.56, 49] # 0.43, 15, 49 negative
    xs = [length/x for x in xs]
#elif flow_unscaled:
#    xs = [21.7e1, 21.7, 21.7e-1, 21.7e-2, 21.7e-3, 21.7e-4, 21.7e-5]
#    peak = [267, 267, 267, 268, 268, 267, 268]
#    ss = [262, 218, 130, 130, 128, 131, 243]
xfine = np.linspace(min(xs), max(xs), 100)

if not peak_log:
    coef = np.polyfit(xs,peak,1)
    poly1d_peak = np.poly1d(coef) 
else:
    coef = np.polyfit(xs,np.log(peak),1)
    poly1d_peak = lambda xs: [np.exp(coef[1]) * np.exp(coef[0] * x) for x in xs]

if not ss_log:
    coef = np.polyfit(xs,ss,2)
    poly1d_ss = np.poly1d(coef) 
else:
    coef = np.polyfit(xs,np.log(ss),1)
    poly1d_ss = lambda xs: [np.exp(coef[1]) * np.exp(coef[0] * x) for x in xs]

# poly1d_fn is now a function which takes in x and returns an estimate for y
print(f'{poly1d_peak = }')
print(f'{poly1d_ss = }')

#plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

if separate_marker_plot:
    plt.plot(xs, peak, marker='.', linestyle='', color='blue')
    plt.plot(xfine, poly1d_peak(xfine), label='Peak Fit', color='blue', linestyle='--')
    plt.plot(xs, ss, marker='x', linestyle='', color='orange')
    plt.plot(xfine, poly1d_ss(xfine), label='Final Fit', color='orange', linestyle='-.')
else:
    plt.plot(xs, peak, marker='.', linestyle='--', color='blue', label='Peak')
    plt.plot(xs, ss, marker='x', linestyle='-.', color='orange', label='Equilibrium')

plt.ylabel('Difference [%]')
plt.legend()
if chem:
    plt.xlabel('Removal Rate [1/s]')
    save_name = 'removal_err'
else:
    #plt.xlabel('Flow Rate [cm/s]')
    plt.xlabel('Residence Time [s]')
    plt.xscale('log')
    plt.yscale('log')
if flow_scaled:
    save_name = 'flow_scaled_err'
#elif flow_unscaled:
#    save_name = 'flow_unscaled_err'

plt.tight_layout()
plt.savefig(f'{save_name}.png')
plt.close()
