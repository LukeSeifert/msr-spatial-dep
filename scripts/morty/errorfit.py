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

#xs = [5, 1, 0.5, 0.2, 0.1, 0.05, 0.03, 0.01, 0.0]
#peak = [93.37, 72.1, 53.6, 27.5, 14.7, 7.5, 4.57, 1.5, 0.0033]
#ss = [93.37, 72.1, 53.4, 23.2, 7.75, 2.0, 0.72, 0.085, 9.4e-6]
xs = [0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
peak = [7.5, 6.07, 4.57, 3.06, 1.5, 0.0033]
ss = [2.0, 1.29, 0.72, 0.32, 0.085, 9.4e-6]
#peak = [15.76, 8.64, 4.92, 3.02, 1.10]
#ss = [4.10, 2.24, 1.71, 1.56, 0.80]

coef = np.polyfit(xs,peak,1)
poly1d_peak = np.poly1d(coef) 
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


plt.plot(xs, peak, marker='.', linestyle='', color='blue')
plt.plot(xs, poly1d_peak(xs), label='Peak Fit', color='blue', linestyle='--')
plt.plot(xs, ss, marker='x', linestyle='', color='orange')
plt.plot(xs, poly1d_ss(xs), label='Final Fit', color='orange', linestyle='--')
plt.ylabel('Error [%]')
plt.legend()
plt.xlabel('Removal Rate [$s^{-1}$]')
plt.tight_layout()
plt.savefig('removal_err.png')
plt.close()
