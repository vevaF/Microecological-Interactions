##############################################################################
# 2D diffusion–advection–reaction model for a multi-species bacterial system
# with competition for space (i.e. movement away from high total densities) 
# and commensalism
#
# State:
#   Pi(x,t): bacterial density for species i = 1,2,3
#   Cj(x,t): substrate (nutrient) density j = 1,2,3
#
# Movement for Pi, i=1,2,3:
#   Drift down the pressure gradient (negative chemotaxis), with pressure
#   Pr = alpha * (P1 + P2 + P3). 
#   Diffusion
#
# Movement for Cj, j=1,2,3:
#   Diffusion
#
# Reaction terms Pi, i=1,2,3 ('netgrowth_i'):
#   Substrate-dependent growth (Monod kinetics) 
#   Density-dependent decay
#
# Reaction terms for Cj, j=1,2,3 ('consumption', 'production'):  
#   Substrate consumption/production by P1, P2, P3, respectively based on Monod kinetics
# Substrate chain (commensalism):
#   - C1 is consumed by P1 and (partly) converted into C2.
#   - C2 is consumed by P2 and (partly) converted into C3.
#   - C3 is consumed by P3.
#
# System for i=1,2,3:
#  dP_i/dt = ∇·( D_i ∇P_i + A_i P_i ∇Pr ) + netgrowth_i(P1, P2, P3, Ci)
#  dC1/dt   = d ΔC1 - consumption(P1, C1) 
#  dC2/dt   = d ΔC2 - consumption(P2, C2) + production(P1,C1)
#  dC3/dt   = d ΔC3 - consumption(P3, C3) + production(P2, C2)
#
# Domain:
#   2D rectangle [0,Lx] x [0,Ly] 
#   with no-flux boundary conditions (homog. Neumann BC) for P1,P2,P3
#   with influx of substrate for C1 (Neumann BC), no-flux BC for C2,C3
#
# Author: V.Freingruber
##############################################################################

# Imports
import numpy as np
import numpy.matlib
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, MaxNLocator
from matplotlib import colors

# Start timer for duration of simulation run
start_time = time.time()

################################################################
# -------------------------- PARAMETERS ----------------------- #
################################################################

global Nx, Ny, Lx, Ly, dx, dy, x, y, x_mp, D, A, d, K1, K2, gam, beta, sig, sigma

# Solver grid
Nx = 100
Ny = 100
Lx = 1
Ly = 1
eps = 10**(-9)

dx = Lx/(Nx - 1)
dy = Ly/(Ny - 1)
y = dy*np.arange(0, Ny, 1)
x = dx*np.arange(0, Nx, 1)

# Motility (diffusion and pressure sensitivity)
D1 = 10**(-6)
D2 = D1
D3 = D1
A1 = 10**(-5)
A2 = A1
A3 = A1

# Substrate diffusivity (same for C1, C2, C3)
d = 10**(-5)

# Reaction parameters
alpha = 1  # pressure scaling
K1 = 1
K2 = 1
K3 = 1
r1 = 1
r2 = 1
r3 = 1
b1 = 0.1
b2 = 0.1
b3 = 0.1
sig = 10     # yield coefficients; in paper: Y1=Y2=Y3 =: sig
sigma = 0.5 # conversion rate of one substrate into another

# Substrate boundary “inflow/outflow” strengths
C1inf = 0
C2inf = 0
C3inf = 0

def initial_conditions_biomass(Nx,Ny,random_seed):
    '''
    Nx, Ny: spatial discretisation variables
    random_seed: integer number for picking a random seed
    
    Function creates initial condition for three populations within a circular support
    '''
    # Spatial grid
    #Nx, Ny = 100, 100
    np.random.seed(random_seed)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Circular support for populations
    main_radius = 0.2
    center = (0.5, 0.5) # centre in the middle of the domain; change if Lx and Ly \neq 1
    distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    support = distance_from_center <= main_radius
    
    # Three components (species) initial pattern: many small circles randomly placed
    total_density = 1.0
    populations = np.zeros((3, Nx, Ny))
    
    # create a number of num_circles small circles with radius small_radius
    # inside support of population
    num_circles = 300
    small_radius = 0.05
    
    for _ in range(num_circles):
        
        # Choose random centre inside main circle
        # such that a circle with radius small_radius is still inside support
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, main_radius - small_radius)
        # centre coordinates
        cx = center[0] + r * np.cos(theta)
        cy = center[1] + r * np.sin(theta)
        # mask for small circle
        small_circle = (X - cx)**2 + (Y - cy)**2 <= small_radius**2
        # Assign disc to a random species
        species_index = np.random.choice(3)
        populations[species_index] += small_circle.astype(float)
    
    # ensure that population density outside of support is 0
    for i in range(3):
        populations[i][~support] = 0
    
    # normalise so summed components equal total_density
    density_sum = np.sum(populations, axis=0)
    # avoid division by 0 
    density_sum[density_sum == 0] = 1
    populations = populations / density_sum * total_density
    
    return populations

# Initial conditions for P1, P2, P3 from the random initial cond. function above
populations = initial_conditions_biomass(Nx,Ny,42)
P10 = populations[0]
P20 = populations[1]
P30 = populations[2]

# Initial substrates: C1 present, C2 and C3 nearly zero
C10 = 5*np.ones([Nx, Ny])
C20 = np.zeros([Nx, Ny])
C30 = np.zeros([Nx, Ny])

# Pressure (filled in RHS)
Pr = np.zeros([Nx, Ny])

################################################################
# ----------------------- RIGHT-HAND SIDE --------------------- #
################################################################
def rhsode(t, y):
    """
    Semi-discrete RHS for the commensalism PDE system (method of lines).
    State vector y stacks P1, P2, P3, C1, C2, C3 on the (Nx × Ny) grid.
    """

    n = 6  # number of fields

    # Flux containers
    fP1 = np.zeros([Nx, Ny, n])
    fP2 = np.zeros([Nx, Ny, n])
    fP3 = np.zeros([Nx, Ny, n])
    fC1 = np.zeros([Nx, Ny, n])
    fC2 = np.zeros([Nx, Ny, n])
    fC3 = np.zeros([Nx, Ny, n])

    # Unpack state
    P1 = np.reshape(y[0:Nx*Ny],            ([Nx, Ny]))
    P2 = np.reshape(y[Nx*Ny:2*Nx*Ny],    ([Nx, Ny]))
    P3 = np.reshape(y[2*Nx*Ny:3*Nx*Ny],  ([Nx, Ny]))
    C1 = np.reshape(y[3*Nx*Ny:4*Nx*Ny],  ([Nx, Ny]))
    C2 = np.reshape(y[4*Nx*Ny:5*Nx*Ny],  ([Nx, Ny]))
    C3 = np.reshape(y[5*Nx*Ny:6*Nx*Ny],  ([Nx, Ny]))

    # Pressure
    Pr = alpha*(P1 + P2 + P3)

    # Substrate fluxes (central differences + simple boundary contributions)
    # C1
    fC1[0:Nx-1, :, 0] = - d * np.diff(C1, axis=0)
    fC1[1:Nx,   :, 1] = fC1[0:Nx-1, :, 0]
    fC1[:, 0:Ny-1, 2] = - d * np.diff(C1, axis=1)
    fC1[:, 1:Ny,   3] = fC1[:, 0:Ny-1, 2]
    fC1[Nx-1, :,   0] = -C1inf * np.ones(Ny)
    fC1[0,      :, 1] =  C1inf * np.ones(Ny)
    fC1[:, Ny-1,   2] = -C1inf * np.ones(Nx)
    fC1[:, 0,      3] =  C1inf * np.ones(Nx)

    # C2
    fC2[0:Nx-1, :, 0] = - d * np.diff(C2, axis=0)
    fC2[1:Nx,   :, 1] = fC2[0:Nx-1, :, 0]
    fC2[:, 0:Ny-1, 2] = - d * np.diff(C2, axis=1)
    fC2[:, 1:Ny,   3] = fC2[:, 0:Ny-1, 2]
    fC2[Nx-1, :,   0] = -C2inf * np.ones(Ny)
    fC2[0,      :, 1] =  C2inf * np.ones(Ny)
    fC2[:, Ny-1,   2] = -C2inf * np.ones(Nx)
    fC2[:, 0,      3] =  C2inf * np.ones(Nx)

    # C3
    fC3[0:Nx-1, :, 0] = - d * np.diff(C3, axis=0)
    fC3[1:Nx,   :, 1] = fC3[0:Nx-1, :, 0]
    fC3[:, 0:Ny-1, 2] = - d * np.diff(C3, axis=1)
    fC3[:, 1:Ny,   3] = fC3[:, 0:Ny-1, 2]
    fC3[Nx-1, :,   0] = -C3inf * np.ones(Ny)
    fC3[0,      :, 1] =  C3inf * np.ones(Ny)
    fC3[:, Ny-1,   2] = -C3inf * np.ones(Nx)
    fC3[:, 0,      3] =  C3inf * np.ones(Nx)

    # Allocate time-derivative arrays
    dP1dt = np.zeros([Nx, Ny])
    dP2dt = np.zeros([Nx, Ny])
    dP3dt = np.zeros([Nx, Ny])
    dC1dt = np.zeros([Nx, Ny])
    dC2dt = np.zeros([Nx, Ny])
    dC3dt = np.zeros([Nx, Ny])

    # Upwind-like split on ∇Pr for drift
    DPrxp = np.where(np.diff(Pr, axis=0) > 0, np.diff(Pr, axis=0), 0)
    DPrxm = np.where(np.diff(Pr, axis=0) < 0, np.diff(Pr, axis=0), 0)
    DPryp = np.where(np.diff(Pr, axis=1) > 0, np.diff(Pr, axis=1), 0)
    DPrym = np.where(np.diff(Pr, axis=1) < 0, np.diff(Pr, axis=1), 0)

    # Bacterial fluxes: diffusion + drift down ∇Pr
    fP1[0:Nx-1, :, 0] = - D1 * np.diff(P1, axis=0) - A1 * (DPrxp * P1[0:Nx-1, :] + DPrxm * P1[1:Nx, :])
    fP1[1:Nx,   :, 1] = fP1[0:Nx-1, :, 0]
    fP1[:, 0:Ny-1, 2] = - D1 * np.diff(P1, axis=1) - A1 * (DPryp * P1[:, 0:Ny-1] + DPrym * P1[:, 1:Ny])
    fP1[:, 1:Ny,   3] = fP1[:, 0:Ny-1, 2]

    fP2[0:Nx-1, :, 0] = - D2 * np.diff(P2, axis=0) - A2 * (DPrxp * P2[0:Nx-1, :] + DPrxm * P2[1:Nx, :])
    fP2[1:Nx,   :, 1] = fP2[0:Nx-1, :, 0]
    fP2[:, 0:Ny-1, 2] = - D2 * np.diff(P2, axis=1) - A2 * (DPryp * P2[:, 0:Ny-1] + DPrym * P2[:, 1:Ny])
    fP2[:, 1:Ny,   3] = fP2[:, 0:Ny-1, 2]

    fP3[0:Nx-1, :, 0] = - D3 * np.diff(P3, axis=0) - A3 * (DPrxp * P3[0:Nx-1, :] + DPrxm * P3[1:Nx, :])
    fP3[1:Nx,   :, 1] = fP3[0:Nx-1, :, 0]
    fP3[:, 0:Ny-1, 2] = - D3 * np.diff(P3, axis=1) - A3 * (DPryp * P3[:, 0:Ny-1] + DPrym * P3[:, 1:Ny])
    fP3[:, 1:Ny,   3] = fP3[:, 0:Ny-1, 2]

    # Transport contributions (divergence of fluxes)
    dP1dt = 1/(dx**2) * (fP1[:, :, 1] - fP1[:, :, 0]) + 1/(dy**2) * (fP1[:, :, 3] - fP1[:, :, 2])
    dP2dt = 1/(dx**2) * (fP2[:, :, 1] - fP2[:, :, 0]) + 1/(dy**2) * (fP2[:, :, 3] - fP2[:, :, 2])
    dP3dt = 1/(dx**2) * (fP3[:, :, 1] - fP3[:, :, 0]) + 1/(dy**2) * (fP3[:, :, 3] - fP3[:, :, 2])
    dC1dt = 1/(dx**2) * (fC1[:, :, 1] - fC1[:, :, 0]) + 1/(dy**2) * (fC1[:, :, 3] - fC1[:, :, 2])
    dC2dt = 1/(dx**2) * (fC2[:, :, 1] - fC2[:, :, 0]) + 1/(dy**2) * (fC2[:, :, 3] - fC2[:, :, 2])
    dC3dt = 1/(dx**2) * (fC3[:, :, 1] - fC3[:, :, 0]) + 1/(dy**2) * (fC3[:, :, 3] - fC3[:, :, 2])

    # Simple transport mass-balance check
    if abs(sum(sum(dP1dt))) > 10**(-6):
        print('The flux is not 0! ' + str(sum(sum(dP1dt))))

    # Substrate kinetics (commensal chain)
    kinetics_C1 = -sig * r1 * P1 * C1 / (K1 + C1 + eps)
    kinetics_C2 =  sig*sigma * r1 * P1 * C1/ (K1 + C1 + eps) - sig * r2 * P2 * C2 / (K2 + C2 + eps)
    kinetics_C3 =  sig*sigma * r2 * P2 * C2/ (K2 + C2 + eps) - sig * r3 * P3 * C3 / (K3 + C3 + eps)

    dC1dt += kinetics_C1
    dC2dt += kinetics_C2
    dC3dt += kinetics_C3

    # Population kinetics: growth (Monod in Ci) and quadratic self-limitation
    dP1dt +=  P1 * (r1 * C1 / (K1 + C1) - b1 * Pr)
    dP2dt +=  P2 * (r2 * C2 / (K2 + C2) - b2 * Pr)
    dP3dt +=  P3 * (r3 * C3 / (K3 + C3) - b3 * Pr)

    # Pack back to vector
    dudt = np.zeros(n * Nx * Ny)
    dudt[0:Nx*Ny]            = dP1dt.flatten()
    dudt[Nx*Ny:2*Nx*Ny]     = dP2dt.flatten()
    dudt[2*Nx*Ny:3*Nx*Ny]   = dP3dt.flatten()
    dudt[3*Nx*Ny:4*Nx*Ny]   = dC1dt.flatten()
    dudt[4*Nx*Ny:5*Nx*Ny]   = dC2dt.flatten()
    dudt[5*Nx*Ny:6*Nx*Ny]   = dC3dt.flatten()
    return dudt

################################################################
# ----------------------- INITIAL STATE ----------------------- #
################################################################

U0 = np.zeros(6 * Nx * Ny)
U0[0:Nx*Ny]                 = P10.flatten()
U0[Nx*Ny:2*Nx*Ny]     = P20.flatten()
U0[2*Nx*Ny:3*Nx*Ny]   = P30.flatten()
U0[3*Nx*Ny:4*Nx*Ny]   = C10.flatten()
U0[4*Nx*Ny:5*Nx*Ny]   = C20.flatten()
U0[5*Nx*Ny:6*Nx*Ny]   = C30.flatten()

################################################################
# -------------------------- SOLVE ---------------------------- #
################################################################

Tend = 500
interval = [0, Tend]
tsave = np.arange(0, Tend + 1, 10)

solution = solve_ivp(rhsode, interval, U0, method='RK45', t_eval=tsave)

# Optional CSV saving (kept commented)
# np.savetxt("test_y.csv", solution.y, delimiter=',')
# np.savetxt("test_t.csv", solution.t, delimiter=',')

# Print duration of simulation in seconds
print('Duration: ' + str(time.time() - start_time) + ' seconds')

################################################################
# -------------------------- PLOTTING ------------------------- #
################################################################

# Define Time steps that are plotted here
# CHANGE AS DESIRED
Tplot = [0,2,5]

# Extract time series arrays from solver
Tdata = solution.t
Ydata = solution.y

# prepare data for plotting
P1 = np.empty((len(Tplot), Ny, Nx))
P2 = np.empty((len(Tplot), Ny, Nx))
P3 = np.empty((len(Tplot), Ny, Nx))
C1 = np.empty((len(Tplot), Ny, Nx))
C2 = np.empty((len(Tplot), Ny, Nx))
C3 = np.empty((len(Tplot), Ny, Nx))

for t in range(len(Tplot)):
    
    P1[t,:,:] = np.reshape(Ydata[0:Nx*Ny,         Tplot[t]], (Ny, Nx))
    P2[t,:,:] = np.reshape(Ydata[Nx*Ny:2*Nx*Ny,   Tplot[t]], (Ny, Nx))
    P3[t,:,:] = np.reshape(Ydata[2*Nx*Ny:3*Nx*Ny, Tplot[t]], (Ny, Nx))
    C1[t,:,:] = np.reshape(Ydata[3*Nx*Ny:4*Nx*Ny, Tplot[t]], (Ny, Nx))
    C2[t,:,:] = np.reshape(Ydata[4*Nx*Ny:5*Nx*Ny, Tplot[t]], (Ny, Nx))
    C3[t,:,:] = np.reshape(Ydata[5*Nx*Ny:6*Nx*Ny, Tplot[t]], (Ny, Nx))

# Adjust settings for plotting:

# global plot styling
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Define colour scales from white to green/orange/purple
from matplotlib.colors import LinearSegmentedColormap
# White→colour scales
cmap_u1 = LinearSegmentedColormap.from_list("white_to_darkgreen",  ["white", "darkgreen"])
cmap_u2 = LinearSegmentedColormap.from_list("white_to_darkorange", ["white", "darkorange"])
cmap_u3 = LinearSegmentedColormap.from_list("white_to_purple",     ["white", "purple"])
cmap_c  = LinearSegmentedColormap.from_list("white_to_blue",       ["white", "royalblue"])

# Definition of two plotting scripts below

def plotting_populations(P1, P2, P3, cmap_u1, cmap_u2, cmap_u3, Tplot):
    
    fig, axes = plt.subplots(3, len(Tplot), figsize=(10, 10), constrained_layout = True)
    fig.set_constrained_layout_pads(w_pad = -10)#w_pad = 0.05, h_pad = 0.05, hspace = 0.2, wspace=0.1)
    #fig.subplots_adjust(hspace = 0.4)
    
    vmax = max(P1.max(),P2.max(),P3.max())
    vP1 = P1.max()
    vP2 = P2.max()
    vP3 = P3.max()
    #norm = Normalize(vmin=0, vmax=1)
    nlevels = 50
    
    for t in range(len(Tplot)):
    
    # Creating contourf plots

        vP1 = P1[t,:,:].max()
        vP2 = P2[t,:,:].max()
        vP3 = P3[t,:,:].max()
        vmax = max(vP1, vP2, vP3)
        
        im2 = axes[0,t].contourf(x, y, P1[t,:,:].T, levels=np.linspace(0, vP1, nlevels+1), vmin=0, vmax = vP1, cmap = cmap_u1) #cmap='Greens')
        im3 = axes[1,t].contourf(x, y, P2[t,:,:].T, levels=np.linspace(0, vP2, nlevels+1), vmin=0, vmax = vP2, cmap = cmap_u2) #cmap='Oranges')
        im4 = axes[2,t].contourf(x, y, P3[t,:,:].T, levels=np.linspace(0, vP3, nlevels+1), vmin=0, vmax = vP3, cmap = cmap_u3) #cmap='Purples')
        
        # Set title of each column to $t=...$
        axes[0,t].set_title(f"$t$ = {Tdata[Tplot[t]]:.2f}")
        
        # Adjust colourbar for Population u1
        cbar2 = fig.colorbar(im2, ax=axes[0,t], location = 'right', shrink = 0.55)#, ticks=[0,0.25,0.5,0.75,1])
        cbar2.formatter = ScalarFormatter(useMathText=True)
        #cbar2.formatter.set_powerlimits((0, 0))
        cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar2.locator = MaxNLocator(nbins=4)
        cbar2.ax.yaxis.set_label_position('left')
        cbar2.set_label(r"$u_1$", rotation = 0, labelpad=20)
        #pos2 = cbar2.ax.get_position()
        #cbar2.ax.set_position([pos2.x0, pos2.y0 + 0.15*pos2.height, pos2.width, 0.7*pos2.height])
        cbar2.update_ticks()
        
        # Adjust colourbar for Population u2
        cbar3 = fig.colorbar(im3, ax=axes[1,t], location = 'right', shrink = 0.55)#, ticks=[0,0.25,0.5,0.75,1])
        cbar3.formatter = ScalarFormatter(useMathText=True)
        #cbar3.formatter.set_powerlimits((0, 0))
        cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar3.locator = MaxNLocator(nbins=4)
        cbar3.ax.yaxis.set_label_position('left')
        cbar3.set_label(r"$u_2$", rotation = 0, labelpad=20)
        #pos3 = cbar3.ax.get_position()
        #cbar3.ax.set_position([pos3.x0, pos3.y0 + 0.15*pos3.height, pos3.width, 0.7*pos3.height])
        cbar3.update_ticks()
        
        # Adjust colourbar for Population u3
        cbar4 = fig.colorbar(im4, ax=axes[2,t], location = 'right', shrink = 0.55)#, ticks=[0,0.25,0.5,0.75,1])
        cbar4.formatter = ScalarFormatter(useMathText=True)
        #cbar4.formatter.set_powerlimits((0, 0))
        cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar4.locator = MaxNLocator(nbins=4)
        cbar4.ax.yaxis.set_label_position('left')
        cbar4.set_label(r"$u_3$", rotation = 0, labelpad=20)
        #pos4 = cbar4.ax.get_position()
        #cbar4.ax.set_position([pos4.x0, pos4.y0 + 0.15*pos4.height, pos4.width, 0.7*pos4.height])
        cbar4.update_ticks()
    
    
    #axes[1, 0].set_ylabel("Population $u_1$")
    #axes[2, 0].set_ylabel("Population $u_2$")
    #axes[3, 0].set_ylabel("Population $u_3$")
    #axes[0, 0].set_ylabel("Substrate c")
    
    # Add y-labels
    axes[1, 0].set_ylabel(r"$y$", rotation=0)
    axes[2, 0].set_ylabel(r"$y$", rotation=0)
    axes[0, 0].set_ylabel(r"$y$", rotation=0)
    axes[1, 0].set_yticks([0, 0.5, 1])
    axes[2, 0].set_yticks([0, 0.5, 1])
    axes[0, 0].set_yticks([0, 0.5, 1])
    
    # Add x-labels
    axes[2, 0].set_xlabel(r"$x$")
    axes[2, 1].set_xlabel(r"$x$")
    axes[2, 2].set_xlabel(r"$x$")
    axes[2, 0].set_xticks([0, 0.5, 1])
    axes[2, 1].set_xticks([0, 0.5, 1])
    axes[2, 2].set_xticks([0, 0.5, 1])
    
    # Remove x- and y-ticks from the inner plots
    for i in (0,1):
        axes[i,0].set_xticks([])
        axes[i,1].set_xticks([])
        axes[i,2].set_xticks([])
        
    for i in (1,2):
        axes[0,i].set_yticks([])
        axes[1,i].set_yticks([])
        axes[2,i].set_yticks([])
    
    # Make sure plots are displayed in quatratic windows    
    for i in (0,1,2):
        for j in (0,1,2):
            axes[j,i].set_aspect('equal', adjustable='box')
    
    # use tight layout for a more compact display
    # may not be compatible with the settings for colourbar
    #plt.tight_layout()
    
    plt.show()
    
def plot_overlap(P1, P2, P3, density_threshold, Tplot):

    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
    from matplotlib import colors
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # --- Adjustable settings ---
    #density_threshold = 0.1   # Mask regions where densities are below this value
    #use_global_scaling = False  # Set True to use the same colour scale for all times
    
    def format_cbar(cbar):
        cbar.formatter = ScalarFormatter(useMathText=True)
        #cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()
    
    
    # --- Plotting loop ---
    for it, col in enumerate(Tplot):
        tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"
    
    
        # Mask low-density regions
        P1_masked = np.ma.masked_where(P1[it].T <= density_threshold, P1[it].T)
        P2_masked = np.ma.masked_where(P2[it].T <= density_threshold, P2[it].T)
        P3_masked = np.ma.masked_where(P3[it].T <= density_threshold, P3[it].T)
       
        vmax = max(P1.max(),P2.max(),P3.max())
        vP1 = P1_masked.max()
        vP2 = P2_masked.max()
        vP3 = P3_masked.max()
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 5), constrained_layout=True)
    
        # Overlay all three populations
        nlevels = 7
        
        im1 = ax.contourf(x, y, P1_masked, levels=np.linspace(density_threshold, vP1, nlevels+1), vmin=density_threshold, vmax = vP1, alpha=0.9, zorder=1, cmap= cmap_u1)#cmap='Greens')
        im2 = ax.contourf(x, y, P2_masked, levels=np.linspace(density_threshold, vP2, nlevels+1), vmin=density_threshold, vmax = vP2, alpha=0.9, zorder=2, cmap= cmap_u2)#cmap='Oranges')
        im3 = ax.contourf(x, y, P3_masked, levels=np.linspace(density_threshold, vP3, nlevels+1), vmin=density_threshold, vmax = vP3, alpha=0.9, zorder=3, cmap= cmap_u3)#cmap='Purples')
    
        # Labels and title
        ax.set_title(tlabel)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        
        import matplotlib.lines as mlines
    
        handles = [
            Line2D([0], [0], marker = 's', linestyle='none', markersize = 10, color='#006400', label=r'$u_1$'),
            Line2D([0], [0], marker = 's', linestyle='none', markersize = 10, color='#ff8c00', label=r'$u_2$'),
            Line2D([0], [0], marker = 's', linestyle='none', markersize = 10, color='#800080', label=r'$u_3$')
        ]
        
        fig.legend(
        handles=handles,
        loc='lower right',          # anchor legend to bottom right of bbox
        bbox_to_anchor=(1.02, 0.75), # (x, y) in figure coordinates
        ncol=3,                     # 3 entries next to each other
        frameon=False,              # frame or no frame
        columnspacing=0.35,          # space between two entries
        handlelength=1.5,
        )
    
    
        # Legend
        # legend_handles = [
        #     Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='darkgreen', label=r"$u_1$"),
        #     Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='darkorange', label=r"$u_2$"),
        #     Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='darkviolet', label=r"$u_3$"),
        # ]
        #ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(2,1), ncol=3, frameon=True)
    
        # --- Single colourbar (for P1 only) ---
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1,0.5,1])
        #cb3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #cb3.set_label(r"$u_3$")
        format_cbar(cb3)
        cb3.update_ticks()
        
        cb2 = fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1, 0.5,1])
        #cb2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #cb2.set_label(r"$u_2$")
        format_cbar(cb2)
        cb2.update_ticks()
        
        cb1 = fig.colorbar(im1, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1,0.5,1])
        #cb1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        #cb1.set_label(r"$u_1$")
        format_cbar(cb1)
        cb1.update_ticks()
        
        ax.set_aspect('equal', adjustable='box')
    
    
    # Show all figures
    plt.show()


# Execute plotting functions

plotting_populations(P1, P2, P3, cmap_u1, cmap_u2, cmap_u3, Tplot)

plot_overlap(P1, P2, P3, 0.06, Tplot)