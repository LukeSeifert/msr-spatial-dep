# Spatially resolved depletion approaches

## One spatial dimension, uniform velocity profile, loosely coupled depletion
For each nuclide $i$:

$$\frac{\partial N_i(z, t)}{\partial t} + \nu_i(z, t) \frac{\partial N_i(z, t)}{\partial z} = S_i(z, t) - \mu_i(z, t) N_i(z, t)$$

where $N$ is the atom density [$\frac{atoms}{cm^3}$], $t$ is time [$s$], $\nu$ is the linear flow rate [$\frac{cm}{s}$], $z$ is the linear distance travelled [$cm$], $S$ is the production, or source, term [$\frac{atoms}{cm^3 s}$], and $\mu$ is the loss term [$s^-1$].

### Approach - MORTY PDE

In order to reduce computational cost, the MORTY PDE approach looks only at isobars and without the advection loosely coupled to depletion.
The loose depletion coupling means that the terms in the equation can be updated as frequently or infrequently as desired. The problem can be run even without a depletion step (though it may be easier to run with depletion).
OpenMC can generate and output the relevant production and loss rates after a depletion step, which can then be fed into the MORTY PDE solver.

The solver uses upwind in space (backwards) and explicit in time finite differencing to generate a solution. At higher dimensions, a finite element approach would be better. The following equations show the finite difference approach.

$$\frac{\partial N_i(z, t)}{\partial t} + \nu_i(z, t) \frac{\partial N_i(z, t)}{\partial z} = S_i(z, t) - \mu_i(z, t) N_i(z, t)$$

Removing $i$ subscript for readability and applying finite differencing in space ($k$) and time ($l$).

$$\frac{N_{k, l+1} - N_{k, l}}{\Delta t} + \nu_{k, l} \frac{N_{k, l} - N_{k-1, l}}{\Delta z} = S_{k, l} - \mu_{k, l} N_{k, l}$$


$$N_{k, l+1} = N_{k, l} + \Delta t \left( S_{k, l} - \mu_{k, l} N_{k, l} - \nu_{k, l} \frac{N_{k, l} - N_{k-1, l}}{\Delta z} \right)$$



MORTY PDE runs this iteratively each time step five times by first updating the source terms for each nuclide in the isobar (since they decay into each other) and then calculating the concentration at that time step.

The level of fidelity of the source and loss terms can vary depending on the fidelity of the neutronics solve. At the lowest level, two regions can be used: in-core (in the neutronics solve) and ex-core (decay only).



## One spatial dimension, uniform velocity profile, tightly coupled predictor depletion

The general equation used in depletion calculations is:

$$\frac{dn}{dt} = An$$

Where $n$ is a vector of nuclide concentrations (~2000-by-1) and $A$ is a square depletion matrix (~2000-by-2000) containing the source and loss terms for each nuclide.

Assuming that $A$ is constant over the time step $h$ (the "predictor" depletion method), Euler's method can be used:

$$n_{l+1} = n_l e^{A_l h}$$

Including a single spatial dimension with advection changes the depletion equation to:

$$\frac{\partial n}{\partial t} + \nu \frac{\partial n}{\partial z} = An$$

Applying finite differencing spatially:

$$\frac{\partial n_k}{\partial t} = A_k n_k - \frac{\nu}{\Delta z} (n_{k-1} - n_k)$$

Grouping $n_k$ together:

$$\frac{\partial n_k}{\partial t} = \left( A_k - \frac{\nu}{\Delta z} \right) n_k + \frac{\nu}{\Delta z} n_{k-1}$$

Generating the advective depletion matrix $\hat{A}$:

$$\hat{A}_k = A_k - \frac{\nu}{\Delta z}$$

Plugging in the advective depletion matrix:

$$\frac{\partial n_k}{\partial t} = \hat{A}_k n_k + \frac{\nu}{\Delta z} n_{k-1}$$

Applying Euler method:

$$n_{k, l+1} = n_{k, l} e^{\hat{A}_{k, l} h} + \frac{\nu_{k, l}}{\Delta z} n_{k-1, l}$$

### Approach - Transfer Rates Method

This method takes the "transfer rates" approach from OpenMC for reprocessing and modifies it for advection use.
This approach is useful because implementation should be fairly rapid, and it will provide reasonably accurate results with >3 materials included.
However, the approach requires many materials to have fine spatial resolution, and the cost scales rapidly as the matrix solve grows with the number of materials (simultaneous solve required).

An example with three materials (1, 2, and 3) is given below. Material 1 flows to material 2, 2 to 3, and 3 to 1.


$$
\frac{d}{dt}
\left(\begin{array}{c} 
n_1\\
n_2\\
n_3
\end{array}\right)
=
\left(\begin{array}{ccc} 
\hat{A}_{1, 1} & 0 & F_{1, 3}\\
F_{2, 1} & \hat{A}_{2, 2} & 0\\
0 & F_{3, 2} & \hat{A}_{3, 3}
\end{array}\right)
\left(\begin{array}{c} 
n_1\\
n_2\\
n_3
\end{array}\right)
$$ 

Where:

$$F_{d, s} = \frac{\nu_{d, s}}{\Delta z},$$

in which $\nu_{d, s}$ is the flow rate from material $s$ to material $d$, and:

$$\hat{A}_{mat, k} = A_{mat, k} - F_{d, s}$$

This is identical to transfer rates, as the $F_{d, s}$ terms are in units of per time, and can thus already be represented in the current version of OpenMC ($d$ represents flow destination, and $s$ is flow source).

Replicating the example from the OpenMC documentation where material 1 flows to material 2:

$$
\frac{d}{dt}
\left(\begin{array}{c} 
n_1\\
n_2
\end{array}\right)
=
\left(\begin{array}{cc} 
\hat{A}_{1, 1} & 0\\
F_{2, 1} & A_{2, 2}
\end{array}\right)
\left(\begin{array}{c} 
n_1\\
n_2
\end{array}\right)
$$ 

This is the exact same form from the OpenMC documentation, except the transfer rate term, $T_{i, j}$, is represented by $F_{d, s}$.


### Approach - PDE Homogeneous Slicing

This method takes the depletion matrix for each material and uses that in the finite differencing format previously described. This approach will initially be a homogeneous approach, where the depletion matrix will be assumed constant over that entire spatial region. In theory, it is possible to use the transfer rates dictionary to reconstruct a flow path and then approximate how materials fit together to have a more accurate representation of the problem spatially. However, that approach will not be used here.

This approach is useful as it should increase the spatial resolution without requiring a large number of materials, which would increase computational cost more.

The homogeneous approximation yields the following simplification:
$$\hat{A}_k = \hat{A}_{material}$$

Therefore, the equation to solve for each time step (and for each spatial node) is:
$$n_{k, l+1} = n_{k, l} e^{\hat{A}_k h} + h \frac{\nu_k}{\Delta z} n_{k-1, l}$$

The first half of the equation can be solved using CRAM. The second half will then need to be added to the result to properly update the concentration.

For numerical stability, the $\lambda_k$ term *must* be less than 1:
$$\lambda_k = \frac{h \nu_k}{\Delta z}$$

The process for solving the problem with this method is as follows:
1. Run transport to generate the depletion matrices for each material
2. Determine time sub-step based on $\lambda_k$ of 0.9 (specific value needs testing), flow rates, number of spatial nodes, and spatial dimensions provided
3. Step through each time step, and within each time step solve spatially
4. Iterate until the time has reached $h$, or the depletion time step (if the sub-step will be too large, use a smaller value)
5. Loop for new depletion time step, or return
