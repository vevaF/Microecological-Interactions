##############################################################################
# 2D diffusion–advection–reaction model for a multi-species bacterial system
# with competition for space (i.e. movement away from high total densities) and a shared nutrient.
#
# State:
#   Pi(x,t): bacterial density for species i = 1,2,3
#   C(x,t)  : substrate (nutrient) density
#
# Movement for Pi, i=1,2,3:
#   Drift down the pressure gradient (negative chemotaxis), with pressure
#   Pr = alpha * (P1 + P2 + P3). 
#   Diffusion
#
# Movement for C:
#   Diffusion
#
# Reaction terms Pi, i=1,2,3 ('netgrowth_i'):
#   Substrate-dependent growth (Monod kinetics) 
#   Density-dependent decay
#
# Reaction terms for C ('çonsumption'):  
#   Substrate consumption by P1, P2, P3 based on Monod kinetics
#
# System for i=1,2,3:
#  dP_i/dt = ∇·( D_i ∇P_i + A_i P_i ∇Pr ) + netgrowth_i(P1, P2, P3 C)
#  dC/dt   = d ΔC - consumption(P1, P2, P3, C)
#
# Domain:
#   2D rectangle [0,Lx] x [0,Ly] 
#   with no-flux boundary conditions (homog. Neumann BC) for P1,P2,P3
#   with influx of substrate for C (Neumann BC)
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

# Timer for duration of simulation run
start_time = time.time()

################################################################
# -------------------------- PARAMETERS ----------------------- #
################################################################

global Nx, Ny, Lx, Ly, dx, dy, x, y, x_mp, D, A, d, K1, K2, gam, beta, sigma

# Solver grid (used by the PDE solver)
Nx = 100
Ny = 100
Lx = 1
Ly = 1

# Small tolerance to avoid division by zero
eps = 10**(-9)

# Grid steps and solver coordinates
dx = Lx/(Nx - 1)
dy = Ly/(Ny - 1)
y = dy*np.arange(0, Ny, 1)
x = dx*np.arange(0, Nx, 1)

# Bacterial motility parameters
D1 = 10**(-6)
D2 = D1
D3 = D1
A1 = 10**(-5)
A2 = A1
A3 = A1

# Substrate diffusivity
d = 10**(-4)

# Reaction parameters
alpha = 1                    # pressure scaling
r1 = 1#0.5
r2 = 1#0.5
r3 = 1#0.5

K1 = 1#20
K2 = 1#20
K3 = 1#20

sig1 = 1/5
sig2 = 1/5
sig3 = 1/5

b1 = 0.1#0.01
b2 = 0.1#0.01
b3 = 0.1#0.01

# Substrate boundary flux strength (used as a simple inflow/outflow setting)
Cinf = 0.00001

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

# Initial substrate C
C0 = 5*np.ones([Nx, Ny])

# Pressure initialisation (filled later in rhs)
Pr = np.zeros([Nx, Ny])

################################################################
# ----------------------- RIGHT-HAND SIDE --------------------- #
################################################################
def rhsode(t, y):
    """
    Semi-discrete RHS for the PDE system (method of lines).
    y is a flattened vector containing P1, P2, P3, C on the solver grid.
    """

    # Flux tensors for P1,P2,P3 and C
    fP1 = np.zeros([Nx, Ny, 4])
    fP2 = np.zeros([Nx, Ny, 4])
    fP3 = np.zeros([Nx, Ny, 4])
    fC  = np.zeros([Nx, Ny, 4])

    # Unpack state
    P1 = np.reshape(y[0:Nx*Ny], ([Nx, Ny]))
    P2 = np.reshape(y[Nx*Ny:2*Nx*Ny], ([Nx, Ny]))
    P3 = np.reshape(y[2*Nx*Ny:3*Nx*Ny], ([Nx, Ny]))
    C  = np.reshape(y[3*Nx*Ny:4*Nx*Ny], ([Nx, Ny]))

    # Total density/ Pressure
    Pr = alpha*(P1 + P2 + P3)

    # Substrate fluxes (central differences with Neumann BC)
    fC[0:Nx-1, :, 0] = - d * np.diff(C, axis=0)
    fC[1:Nx,   :, 1] = fC[0:Nx-1, :, 0]
    fC[:, 0:Ny-1, 2] = - d * np.diff(C, axis=1)
    fC[:,   1:Ny, 3] = fC[:, 0:Ny-1, 2]

    # Boundary contributions (in/outflow proxies)
    fC[Nx-1, :, 0] = -Cinf * np.ones(Ny)  # right
    fC[0,    :, 1] =  Cinf * np.ones(Ny)  # left
    fC[:, Ny-1, 2] = -Cinf * np.ones(Nx)  # top
    fC[:,    0, 3] =  Cinf * np.ones(Nx)  # bottom

    # Prepare arrays for time derivatives
    dP1dt = np.zeros([Nx, Ny])
    dP2dt = np.zeros([Nx, Ny])
    dP3dt = np.zeros([Nx, Ny])
    dCdt  = np.zeros([Nx, Ny])

    # Upwind-like split of pressure gradients (for drift direction)
    DPrxp = np.where(np.diff(Pr, axis=0) > 0, np.diff(Pr, axis=0), 0)
    DPrxm = np.where(np.diff(Pr, axis=0) < 0, np.diff(Pr, axis=0), 0)
    DPryp = np.where(np.diff(Pr, axis=1) > 0, np.diff(Pr, axis=1), 0)
    DPrym = np.where(np.diff(Pr, axis=1) < 0, np.diff(Pr, axis=1), 0)

    # Bacterial fluxes (diffusion + drift down ∇Pr)
    fP1[0:Nx-1, :, 0] = - D1 * np.diff(P1, axis=0) - A1 * (DPrxp * P1[0:Nx-1, :] + DPrxm * P1[1:Nx, :])
    fP1[1:Nx,   :, 1] = fP1[0:Nx-1, :, 0]
    fP1[:, 0:Ny-1, 2] = - D1 * np.diff(P1, axis=1) - A1 * (DPryp * P1[:, 0:Ny-1] + DPrym * P1[:, 1:Ny])
    fP1[:,   1:Ny, 3] = fP1[:, 0:Ny-1, 2]

    fP2[0:Nx-1, :, 0] = - D2 * np.diff(P2, axis=0) - A2 * (DPrxp * P2[0:Nx-1, :] + DPrxm * P2[1:Nx, :])
    fP2[1:Nx,   :, 1] = fP2[0:Nx-1, :, 0]
    fP2[:, 0:Ny-1, 2] = - D2 * np.diff(P2, axis=1) - A2 * (DPryp * P2[:, 0:Ny-1] + DPrym * P2[:, 1:Ny])
    fP2[:,   1:Ny, 3] = fP2[:, 0:Ny-1, 2]

    fP3[0:Nx-1, :, 0] = - D3 * np.diff(P3, axis=0) - A3 * (DPrxp * P3[0:Nx-1, :] + DPrxm * P3[1:Nx, :])
    fP3[1:Nx,   :, 1] = fP3[0:Nx-1, :, 0]
    fP3[:, 0:Ny-1, 2] = - D3 * np.diff(P3, axis=1) - A3 * (DPryp * P3[:, 0:Ny-1] + DPrym * P3[:, 1:Ny])
    fP3[:,   1:Ny, 3] = fP3[:, 0:Ny-1, 2]

    # Divergence of fluxes
    dP1dt = 1/(dx**2) * (fP1[:, :, 1] - fP1[:, :, 0]) + 1/(dy**2) * (fP1[:, :, 3] - fP1[:, :, 2])
    dP2dt = 1/(dx**2) * (fP2[:, :, 1] - fP2[:, :, 0]) + 1/(dy**2) * (fP2[:, :, 3] - fP2[:, :, 2])
    dP3dt = 1/(dx**2) * (fP3[:, :, 1] - fP3[:, :, 0]) + 1/(dy**2) * (fP3[:, :, 3] - fP3[:, :, 2])
    dCdt  = 1/(dx**2) * (fC[:,  :, 1] - fC[:,  :, 0]) + 1/(dy**2) * (fC[:,  :, 3] - fC[:,  :, 2])

    # Check for numerical mass conservation (for one population) of transport
    if abs(sum(sum(dP1dt))) > 10**(-6):
        print('The flux is not 0! ' + str(sum(sum(dP1dt))))

    # Substrate kinetics (Monod-type uptake by each species)
    kinetics_lig = - ( r1/sig1 * P1 * C/(K1 + C + eps)
                     + r2/sig2 * P2 * C/(K2 + C + eps)
                     + r3/sig3 * P3 * C/(K2 + C + eps) )
    dCdt = dCdt + kinetics_lig

    # Population kinetics (growth with Monod kinetics, density dependent decay)
    dP1dt += P1 * (r1 * C/(K1 + C) - b1 * Pr)
    dP2dt += P2 * (r2 * C/(K2 + C) - b2 * Pr)
    dP3dt += P3 * (r3 * C/(K3 + C) - b3 * Pr)

    # Pack RHS into a vector
    dudt = np.zeros(4 * Nx * Ny)
    dudt[0:Nx*Ny]           = dP1dt.flatten()
    dudt[Nx*Ny:2*Nx*Ny]     = dP2dt.flatten()
    dudt[2*Nx*Ny:3*Nx*Ny]   = dP3dt.flatten()
    dudt[3*Nx*Ny:4*Nx*Ny]   = dCdt.flatten()
    
    return dudt

################################################################
# ----------------------- INITIAL STATE ----------------------- #
################################################################

# Stack initial data into a single vector U0 = [P1, P2, P3, C]
U0 = np.zeros(4 * Nx* Ny)
U0[0:Nx*Ny]           = P10.flatten()
U0[Nx*Ny:2*Nx*Ny]     = P20.flatten()
U0[2*Nx*Ny:3*Nx*Ny]   = P30.flatten()
U0[3*Nx*Ny:4*Nx*Ny]   = C0.flatten()

################################################################
# -------------------------- SOLVE ---------------------------- #
################################################################

Tend = 200#500
interval = [0, Tend]
tsave = np.arange(0, Tend + 1, 1)

solution = solve_ivp(rhsode, interval, U0, method='RK45', t_eval=tsave)

# Optional CSV saving; uncomment below to save results as csv file
# np.savetxt("test_y.csv", solution.y, delimiter=',')
# np.savetxt("test_t.csv", solution.t, delimiter=',')

# Print duration of simulation in seconds
print('Duration: ' + str(time.time() - start_time) + ' seconds')

################################################################
# -------------------------- PLOTTING ------------------------- #
################################################################

# PREPARATION OF PLOTTING DATA 

# Times to plot
# ENTER HERE THE DESIRED PLOTTING TIMES
Tplot = [0,25,50]

# Extract time series arrays from solver
Tdata = solution.t
Ydata = solution.y

# Prepare data for plotting
P1 = np.empty((len(Tplot),Ny,Nx))
P2 = np.empty((len(Tplot),Ny,Nx))
P3 = np.empty((len(Tplot),Ny,Nx))
C = np.empty((len(Tplot),Ny,Nx))

for t in range(len(Tplot)):
    
    P1[t,:,:] = np.reshape(Ydata[0:Nx*Ny,           Tplot[t]], (Ny, Nx))
    P2[t,:,:] = np.reshape(Ydata[Nx*Ny:2*Nx*Ny,     Tplot[t]], (Ny, Nx))
    P3[t,:,:] = np.reshape(Ydata[2*Nx*Ny:3*Nx*Ny,   Tplot[t]], (Ny, Nx))
    C[t,:,:]  = np.reshape(Ydata[3*Nx*Ny:4*Nx*Ny,   Tplot[t]], (Ny, Nx))

# Adjust settings for plotting:
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

# DEFINITION OF THREE PLOTTING SCRIPTS BELOW

def plotting_everything(C, P1, P2, P3, Tplot):
    '''
    Plotting all three bacterial populations and the substrate 
    at different timesteps specified in Tplot next to one another
    '''
    
    fig, axes = plt.subplots(4, len(Tplot), figsize=(10, 10), constrained_layout = True)
    
    #vmax = max(P1.max(),P2.max(),P3.max())
    vP1 = P1.max()
    vP2 = P2.max()
    vP3 = P3.max()
    
    for t in range(len(Tplot)):
    
    # Creating contourf plots
        nlevels = 50
        im1 = axes[0, t].contourf(x, y, C[t,:,:].T, levels=nlevels, cmap = cmap_c) #cmap='Blues')
        im2 = axes[1,t].contourf(x, y, P1[t,:,:].T, levels=nlevels, vmin=0, vmax = vP1, cmap = cmap_u1) #cmap='Greens')
        im3 = axes[2,t].contourf(x, y, P2[t,:,:].T, levels=nlevels, vmin=0, vmax = vP2, cmap = cmap_u2) #cmap='Oranges')
        im4 = axes[3,t].contourf(x, y, P3[t,:,:].T, levels=nlevels, vmin=0, vmax = vP3, cmap = cmap_u3) #cmap='Purples')
        
        # Set title of each column to $t=...$
        axes[0,t].set_title(f"$t$ = {Tplot[t]:.2f}")
        
    
    # Adjust colourbar for Substrate c
    cbar1 = fig.colorbar(im1, ax = axes[0,:], location = 'right', pad = 0.055)#pad = 0.08)
    cbar1.formatter = ScalarFormatter(useMathText=True)
    #cbar1.formatter.set_powerlimits((0, 0))
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar1.locator = MaxNLocator(nbins=4)
    cbar1.ax.yaxis.set_label_position('left')
    cbar1.set_label(r"$c$", rotation = 0, labelpad=20)
    cbar1.update_ticks()
    
    # Adjust colourbar for Population u1
    cbar2 = fig.colorbar(im2, ax=axes[1,:], location = 'right')
    cbar2.formatter = ScalarFormatter(useMathText=True)
    #cbar2.formatter.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar2.locator = MaxNLocator(nbins=4)
    cbar2.update_ticks()
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.set_label(r"$u_1$", rotation = 0, labelpad=20)
    
    # Adjust colourbar for Population u2
    cbar3 = fig.colorbar(im3, ax=axes[2,:], location = 'right')
    cbar3.formatter = ScalarFormatter(useMathText=True)
    #cbar3.formatter.set_powerlimits((0, 0))
    cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar3.locator = MaxNLocator(nbins=4)
    cbar3.update_ticks()
    cbar3.ax.yaxis.set_label_position('left')
    cbar3.set_label(r"$u_2$", rotation = 0, labelpad=20)
    
    # Adjust colourbar for Population u3
    cbar4 = fig.colorbar(im4, ax=axes[3,:], location = 'right')
    cbar4.formatter = ScalarFormatter(useMathText=True)
    #cbar4.formatter.set_powerlimits((0, 0))
    cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar4.locator = MaxNLocator(nbins=4)
    cbar4.update_ticks()
    cbar4.ax.yaxis.set_label_position('left')
    cbar4.set_label(r"$u_3$", rotation = 0, labelpad=20)
    
    # Add y-labels
    axes[1, 0].set_ylabel(r"$y$", rotation=0)
    axes[2, 0].set_ylabel(r"$y$", rotation=0)
    axes[3, 0].set_ylabel(r"$y$", rotation=0)
    axes[0, 0].set_ylabel(r"$y$", rotation=0)
    axes[1, 0].set_yticks([0, 0.5, 1])
    axes[2, 0].set_yticks([0, 0.5, 1])
    axes[3, 0].set_yticks([0, 0.5, 1])
    axes[0, 0].set_yticks([0, 0.5, 1])
    
    # Add x-labels
    axes[3, 0].set_xlabel(r"$x$")
    axes[3, 1].set_xlabel(r"$x$")
    axes[3, 2].set_xlabel(r"$x$")
    axes[3, 0].set_xticks([0, 0.5, 1])
    axes[3, 1].set_xticks([0, 0.5, 1])
    axes[3, 2].set_xticks([0, 0.5, 1])
    
    # Remove x- and y-ticks from the inner plots
    for i in (0,1,2):
        axes[i,0].set_xticks([])
        axes[i,1].set_xticks([])
        axes[i,2].set_xticks([])
        
    for i in (1,2):
        axes[0,i].set_yticks([])
        axes[1,i].set_yticks([])
        axes[2,i].set_yticks([])
        axes[3,i].set_yticks([])
    
    # Make sure plots are displayed in quatratic windows    
    for i in (0,1,2):
        for j in (0,1,2,3):
            axes[j,i].set_aspect('equal', adjustable='box')
    
    # use tight layout for a more compact display
    # may not be compatible with the settings for colourbar
    #plt.tight_layout()
    
    plt.show()

def plotting_populations(P1, P2, P3, Tplot):
    '''
    Plotting all three bacterial populations at different timesteps next to one another
    '''
    
    fig, axes = plt.subplots(3, len(Tplot), figsize=(10, 10), constrained_layout = True)
    fig.set_constrained_layout_pads(w_pad = 0.00001)#w_pad = 0.05, h_pad = 0.05, hspace = 0.2, wspace=0.1)
        
    vmax = max(P1.max(),P2.max(),P3.max())
    vP1 = P1.max()
    vP2 = P2.max()
    vP3 = P3.max()
        
    for t in range(len(Tplot)):
    
    # Creating contourf plots
        nlevels = 50
        im2 = axes[0,t].contourf(x, y, P1[t,:,:].T, levels=np.linspace(0, vP1, nlevels+1), vmin=0, vmax = vP1, cmap = cmap_u1) #cmap='Greens')
        im3 = axes[1,t].contourf(x, y, P2[t,:,:].T, levels=np.linspace(0, vP2, nlevels+1), vmin=0, vmax = vP2, cmap = cmap_u2) #cmap='Oranges')
        im4 = axes[2,t].contourf(x, y, P3[t,:,:].T, levels=np.linspace(0, vP3, nlevels+1), vmin=0, vmax = vP3, cmap = cmap_u3) #cmap='Purples')
        
        # Set title of each column to $t=...$
        axes[0,t].set_title(f"$t$ = {Tplot[t]:.2f}")
        
    # Adjust colourbar for Population u1
    cbar2 = fig.colorbar(im2, ax=axes[0,:], location = 'right', shrink = 0.75, ticks=[0,0.25,0.5,0.75,1])
    cbar2.formatter = ScalarFormatter(useMathText=True)
    #cbar2.formatter.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #cbar2.locator = MaxNLocator(nbins=4)
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.set_label(r"$u_1$", rotation = 0, labelpad=20)
    cbar2.update_ticks()
    
    # Adjust colourbar for Population u2
    cbar3 = fig.colorbar(im3, ax=axes[1,:], location = 'right', shrink = 0.75, ticks=[0,0.25,0.5,0.75,1])
    cbar3.formatter = ScalarFormatter(useMathText=True)
    #cbar3.formatter.set_powerlimits((0, 0))
    cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #cbar3.locator = MaxNLocator(nbins=4)
    cbar3.ax.yaxis.set_label_position('left')
    cbar3.set_label(r"$u_2$", rotation = 0, labelpad=20)
    cbar3.update_ticks()
    
    # Adjust colourbar for Population u3
    cbar4 = fig.colorbar(im4, ax=axes[2,:], location = 'right', shrink = 0.75, ticks=[0,0.25,0.5,0.75,1])
    cbar4.formatter = ScalarFormatter(useMathText=True)
    #cbar4.formatter.set_powerlimits((0, 0))
    cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #cbar4.locator = MaxNLocator(nbins=4)
    cbar4.ax.yaxis.set_label_position('left')
    cbar4.set_label(r"$u_3$", rotation = 0, labelpad=20)
    cbar4.update_ticks()
    
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
    '''
    Plotting environment to plot the three population densities on top of each other
    Densities below the density_threshold are cut off to avoid the overlap of low densities on top of each other
    '''

    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
    from matplotlib import colors
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    #density_threshold = 0.1   # passed as an input to the function
    
    def format_cbar(cbar):
        cbar.formatter = ScalarFormatter(useMathText=True)
        #cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()
    
    
    # --- Plotting loop ---
    for it, col in enumerate(Tplot):
        tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"
    
    
        # Mask low-density regions
        P1_masked = np.ma.masked_where(P1[it].T <= density_threshold, P1[it].T)
        P2_masked = np.ma.masked_where(P2[it].T <= density_threshold, P2[it].T)
        P3_masked = np.ma.masked_where(P3[it].T <= density_threshold, P3[it].T)
       
        #vmax = max(P1.max(),P2.max(),P3.max())
        vP1 = P1.max()
        vP2 = P2.max()
        vP3 = P3.max()
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 5), constrained_layout=True)
    
        # choose number of levels in countour plot
        nlevels = 7
        
        #create contourplot
        im1 = ax.contourf(x, y, P1_masked, levels=np.linspace(density_threshold, vP1, nlevels+1), vmin=density_threshold, vmax = vP1, alpha=0.9, zorder=2, cmap= cmap_u1)#cmap='Greens')
        im2 = ax.contourf(x, y, P2_masked, levels=np.linspace(density_threshold, vP1, nlevels+1), vmin=density_threshold, vmax = vP2, alpha=0.9, zorder=1, cmap= cmap_u2)#cmap='Oranges')
        im3 = ax.contourf(x, y, P3_masked, levels=np.linspace(density_threshold, vP1, nlevels+1), vmin=density_threshold, vmax = vP3, alpha=0.9, zorder=3, cmap= cmap_u3)#cmap='Purples')
    
        # Labels and title
        ax.set_title(tlabel)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        
        # legend settings    
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
    
    
        # adding colourbars with optional label, adjust spacing for inclusion of labels
        cb3 = fig.colorbar(im3, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1,0.5,1])
        #cb3.set_label(r"$u_3$")
        format_cbar(cb3)
        
        cb2 = fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1, 0.5,1])
        #cb2.set_label(r"$u_2$")
        format_cbar(cb2)
        
        cb1 = fig.colorbar(im1, ax=ax, fraction=0.04, pad=0.04)#, ticks= [0.1,0.5,1])
        #cb1.set_label(r"$u_1$")
        format_cbar(cb1)
        
        # make sure all plots are displayed in quadratic boxes
        ax.set_aspect('equal', adjustable='box')
    
    
    # Show all figures
    plt.show()


# EXECUTION OF PLOTTING FUNCTIONS
# comment out to avoid plotting

plotting_populations(P1, P2, P3, Tplot)
plotting_everything(C, P1, P2, P3, Tplot)
plot_overlap(P1, P2, P3, 0.1, Tplot)