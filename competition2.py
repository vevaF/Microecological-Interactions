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
# Reaction terms for C ('consumption'):  
#   Substrate consumption by P1, P2, P3 based on Monod kinetics
#
# System for i=1,2:
#  dP_i/dt = ∇·( D_i ∇P_i + A_i P_i ∇Pr ) + netgrowth_i(P1, P2, C)
#  dC/dt   = d ΔC - consumption(P1, P2, C)
#
# Domain:
#   2D rectangle [0,Lx] x [0,Ly] 
#   with no-flux boundary conditions (homog. Neumann BC) for P1,P2,P3
#   with influx of substrate for C (Neumann BC)
#
# Author: V.Freingruber
##############################################################################

# import packages
from __future__ import annotations
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
from matplotlib import colors


# Timer for duration of simulation run
start_time = time.time()

################################################################
# -----------------------Parameters --------------------- #
################################################################

# Grid sizes
Nx = 100            
Ny = 100

# Domain lengths
Lx = 1.0
Ly = 1.0

# Discretisation
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = dx * np.arange(0, Nx)
y = dy * np.arange(0, Ny)

eps = 1e-9

# Transport parameters (populations)
D1 = 1e-6
D2 = D1
D3 = D1
A1 = 1e-5
A2 = A1
A3 = A1

# Substrate diffusion
d = 1e-5
# If you want to include the effective diffusion coefficient inside the biomass 
# uncomment d_eff and 
# use the alternative flux (commented in rhsode).
#d_eff = 0.6 * d

# Pressure and kinetics
alpha = 1.0
r1, r2, r3 = 0.5, 1.0, 0.0
K1, K2, K3 = 100.0, 100.0, 0.0
sig1, sig2, sig3 = 4.0, 0.5, 0.2 # Yield coefficients Y1, Y2, Y3 in paper
b1 = b2 = b3 = 0.0

# Boundary conditions for C at the top boundary
Cinf = 1.0
# Initial bulk concentration for C
C_bulk = 5.0

# Simulation time grid
T_end = 100.0
save_times = np.arange(0.0, T_end + 1.0, 1.0)

################################################################
# ----------------------- Initial conditions --------------------- #
################################################################

def make_initial_populations(pattern):
    """Return (P10, P20) on shape (Nx, Ny) based on the chosen pattern.

    Patterns:
      - "blocks": two adjacent blocks near the bottom.
      - "stripes": alternating vertical stripes (limited to middle width).
    """
    W, H = Nx, Ny
    P10 = np.zeros((W, H))
    P20 = np.zeros((W, H))

    if pattern == "blocks":
        h_pop = int(0.1 * H)        # 10% height
        w_pop = int(0.5 * W)        # 50% centred width
        x_start = (W - w_pop) // 2
        x_end = x_start + w_pop
        mid_x = (x_start + x_end) // 2
        P10[x_start:mid_x, :h_pop] = 1.0
        P20[mid_x:x_end, :h_pop] = 1.0

    elif pattern == "stripes":
        h_pop = int(0.1 * H)
        max_occupied_width_ratio = 0.5
        occupied_width = int(W * max_occupied_width_ratio)
        left_margin = (W - occupied_width) // 2
        right_margin = left_margin + occupied_width
        stripe_width = 15
        for xi in range(left_margin, right_margin, 2 * stripe_width):
            P10[xi:xi + stripe_width, :h_pop] = 1.0
            P20[xi + stripe_width:xi + 2 * stripe_width, :h_pop] = 1.0

    else:
        raise ValueError("Unknown pattern. Use 'blocks' or 'stripes'.")

    return P10, P20


# Choose your initialisation pattern here, uncomment one of these
pattern = "blocks"
pattern = "stripes"
P10, P20 = make_initial_populations(pattern)

# This code is implemented for three populations - see competition3.py 
# Here, third population is set to 0 with 0 kinetics
P30 = np.zeros((Nx, Ny))

# Substrate initial condition
C0 = C_bulk * np.ones((Nx, Ny))

# Pack initial data for solve_ivp (flattened order: P1, P2, P3, C)
U0 = np.zeros(4 * Nx * Ny)
U0[0:Nx * Ny] = P10.flatten()
U0[Nx * Ny:2 * Nx * Ny] = P20.flatten()
U0[2 * Nx * Ny:3 * Nx * Ny] = P30.flatten()
U0[3 * Nx * Ny:4 * Nx * Ny] = C0.flatten()

################################################################
# ----------------------- RIGHT-HAND SIDE --------------------- #
################################################################

def rhsode(t: float, y: np.ndarray) -> np.ndarray:
    """Right-hand side for method of lines.

    y packs [P1, P2, P3, C] each of shape (Nx, Ny), flattened in C order.
    Returns dudt in the same packed format.
    """
    # Unpack state
    P1 = np.reshape(y[0:Nx * Ny], (Nx, Ny))
    P2 = np.reshape(y[Nx * Ny:2 * Nx * Ny], (Nx, Ny))
    P3 = np.reshape(y[2 * Nx * Ny:3 * Nx * Ny], (Nx, Ny))
    C  = np.reshape(y[3 * Nx * Ny:4 * Nx * Ny], (Nx, Ny))

    Pr = alpha * (P1 + P2 + P3)

    # Flux arrays: [:, :, 0/1] for x±, [:, :, 2/3] for y±
    fP1 = np.zeros((Nx, Ny, 4))
    fP2 = np.zeros((Nx, Ny, 4))
    fP3 = np.zeros((Nx, Ny, 4))
    fC  = np.zeros((Nx, Ny, 4))

    # Substrate flux (uniform diffusion). If using crowding-dependent diffusion,
    # switch to the alternative commented lines below.
    fC[0:Nx - 1, :, 0] = -d * np.diff(C, axis=0)
    fC[1:Nx, :, 1]     =  fC[0:Nx - 1, :, 0]
    fC[:, 0:Ny - 1, 2] = -d * np.diff(C, axis=1)
    fC[:, 1:Ny, 3]     =  fC[:, 0:Ny - 1, 2]

    fC[Nx - 1, :, 0] = 0.0
    fC[0, :, 1]      = 0.0
    fC[:, Ny - 1, 2] = -d * Cinf
    fC[:, 0, 3]      = 0.0

    # --- Alternative crowding-dependent diffusion for C ---
    # threshold = 0.01
    # fC[0:Nx-1,:,0] = -(d * (Pr[0:Nx-1,:] < threshold) + d_eff * (Pr[0:Nx-1,:] >= threshold)) * np.diff(C,axis=0)
    # fC[1:Nx,:,1]   = fC[0:Nx-1,:,0]
    # fC[:,0:Ny-1,2] = -(d * (Pr[:,0:Ny-1] < threshold) + d_eff * (Pr[:,0:Ny-1] >= threshold)) * np.diff(C,axis=1)
    # fC[:,1:Ny,3]   = fC[:,0:Ny-1,2]

    # Pressure gradients (upwind split)
    DPrx = np.diff(Pr, axis=0)
    DPry = np.diff(Pr, axis=1)
    DPrxp = np.where(DPrx > 0, DPrx, 0.0)
    DPrxm = np.where(DPrx < 0, DPrx, 0.0)
    DPryp = np.where(DPry > 0, DPry, 0.0)
    DPrym = np.where(DPry < 0, DPry, 0.0)

    # Population fluxes
    fP1[0:Nx - 1, :, 0] = -D1 * np.diff(P1, axis=0) - A1 * (DPrxp * P1[0:Nx - 1, :] + DPrxm * P1[1:Nx, :])
    fP1[1:Nx, :, 1]     =  fP1[0:Nx - 1, :, 0]
    fP1[:, 0:Ny - 1, 2] = -D1 * np.diff(P1, axis=1) - A1 * (DPryp * P1[:, 0:Ny - 1] + DPrym * P1[:, 1:Ny])
    fP1[:, 1:Ny, 3]     =  fP1[:, 0:Ny - 1, 2]

    fP2[0:Nx - 1, :, 0] = -D2 * np.diff(P2, axis=0) - A2 * (DPrxp * P2[0:Nx - 1, :] + DPrxm * P2[1:Nx, :])
    fP2[1:Nx, :, 1]     =  fP2[0:Nx - 1, :, 0]
    fP2[:, 0:Ny - 1, 2] = -D2 * np.diff(P2, axis=1) - A2 * (DPryp * P2[:, 0:Ny - 1] + DPrym * P2[:, 1:Ny])
    fP2[:, 1:Ny, 3]     =  fP2[:, 0:Ny - 1, 2]

    fP3[0:Nx - 1, :, 0] = -D3 * np.diff(P3, axis=0) - A3 * (DPrxp * P3[0:Nx - 1, :] + DPrxm * P3[1:Nx, :])
    fP3[1:Nx, :, 1]     =  fP3[0:Nx - 1, :, 0]
    fP3[:, 0:Ny - 1, 2] = -D3 * np.diff(P3, axis=1) - A3 * (DPryp * P3[:, 0:Ny - 1] + DPrym * P3[:, 1:Ny])
    fP3[:, 1:Ny, 3]     =  fP3[:, 0:Ny - 1, 2]

    # Divergences
    dP1dt = (fP1[:, :, 1] - fP1[:, :, 0]) / (dx ** 2) + (fP1[:, :, 3] - fP1[:, :, 2]) / (dy ** 2)
    dP2dt = (fP2[:, :, 1] - fP2[:, :, 0]) / (dx ** 2) + (fP2[:, :, 3] - fP2[:, :, 2]) / (dy ** 2)
    dP3dt = (fP3[:, :, 1] - fP3[:, :, 0]) / (dx ** 2) + (fP3[:, :, 3] - fP3[:, :, 2]) / (dy ** 2)
    dCdt  = (fC[:,  :, 1] - fC[:,  :, 0])  / (dx ** 2) + (fC[:,  :, 3] - fC[:,  :, 2])  / (dy ** 2)

    # Kinetics
    consumption = -(r1 / sig1) * P1 * C / (K1 + C + eps)-(r2 / sig2) * P2 * C / (K2 + C + eps) 
    dCdt += consumption

    totalP = P1 + P2 + P3
    dP1dt += P1 * (r1 * C / (K1 + C + eps) - b1 * totalP)
    dP2dt += P2 * (r2 * C / (K2 + C + eps) - b2 * totalP)
    dP3dt += 0

    # Pack back
    dudt = np.zeros(4 * Nx * Ny)
    dudt[0:Nx * Ny]                 = dP1dt.flatten()
    dudt[Nx * Ny:2 * Nx * Ny]       = dP2dt.flatten()
    dudt[2 * Nx * Ny:3 * Nx * Ny]   = dP3dt.flatten()
    dudt[3 * Nx * Ny:4 * Nx * Ny]   = dCdt.flatten()
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

def plot_total_mass(Tdata, Ydata, dx):
    """Plot total mass of P1 and P2 over time."""
    P1tot = np.zeros_like(Tdata)
    P2tot = np.zeros_like(Tdata)
    for k in range(len(Tdata)):
        P1 = np.reshape(Ydata[0:Nx * Ny, k], (Ny, Nx))
        P2 = np.reshape(Ydata[Nx * Ny:2 * Nx * Ny, k], (Ny, Nx))
        P1tot[k] = np.sum(P1)
        P2tot[k] = np.sum(P2)
    
    scale = dx**2
    plt.figure(figsize=(6, 4))
    plt.plot(Tdata, P1tot * scale, color='green' , label=r"$M_1$ (YS)")
    plt.plot(Tdata, P2tot * scale, linestyle="-.", color='orange', label=r"$M_2$ (GS)")
    plt.legend()
    plt.title("Total mass of $u_1$, $u_2$ over time")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$M_1, M_2$")
    plt.tight_layout()
    #plt.savefig("total_mass.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_everything(Tdata, Ydata, Tplot):
    """Contour plots of P1, P2, C at selected time indices in `times_to_plot`."""
    
    P1 = np.empty((len(Tplot), Ny, Nx))
    P2 = np.empty((len(Tplot), Ny, Nx))
    C  = np.empty((len(Tplot), Ny, Nx))

    for i, ti in enumerate(Tplot):
        P1[i, :, :] = np.reshape(Ydata[0 * Nx * Ny:1 * Nx * Ny, ti], (Ny, Nx))
        P2[i, :, :] = np.reshape(Ydata[1 * Nx * Ny:2 * Nx * Ny, ti], (Ny, Nx))
        C[i, :, :]  = np.reshape(Ydata[3 * Nx * Ny:4 * Nx * Ny, ti], (Ny, Nx))

    fig, axes = plt.subplots(3, len(Tplot), figsize=(10, 8), constrained_layout=True)
    
    vminP = min(P1.min(), P2.min())
    vmaxP = max(P1.max(), P2.max())
    normP = colors.Normalize(vmin=vminP, vmax=vmaxP)
    normC = colors.Normalize(vmin=C.min(), vmax=C.max())

    for j in range(len(Tplot)):
        im1 = axes[0, j].contourf(x, y, P1[j, :, :].T, levels=50, norm=normP, cmap = cmap_u1)#cmap="Greens")
        im2 = axes[1, j].contourf(x, y, P2[j, :, :].T, levels=50, norm=normP, cmap = cmap_u2)#cmap="Oranges")
        im3 = axes[2, j].contourf(x, y, C[j, :, :].T,  levels=50, norm=normC, cmap = cmap_c)#cmap="Blues"
        axes[0, j].set_title(f"t = {Tplot[j]:.0f}")

    cbar1 = fig.colorbar(im1, ax=axes[0, :], location="right")
    cbar2 = fig.colorbar(im2, ax=axes[1, :], location="right")
    cbar3 = fig.colorbar(im3, ax=axes[2, :], location="right")
    for cb in (cbar1, cbar2, cbar3):
        cb.formatter = ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        cb.locator = MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.ax.yaxis.set_label_position('left')
    cbar1.set_label(r"$u_1$", rotation=0, labelpad=20)
    cbar2.set_label(r"$u_2$", rotation=0, labelpad=20)
    cbar3.set_label(r"$u_3$", rotation=0, labelpad=20)
    
    # set y labels
    axes[0, 0].set_ylabel(r"$y$", rotation=0)
    axes[1, 0].set_ylabel(r"$y$", rotation=0)
    axes[2, 0].set_ylabel(r"$y$", rotation=0)
    # set x labels
    axes[2, 0].set_xlabel(r"$x$")
    axes[2, 1].set_xlabel(r"$x$")
    axes[2, 2].set_xlabel(r"$x$")
    
    # Simplify ticks
    for row in range(3):
        for col in range(len(Tplot)):
            if row < 2:
                axes[row, col].set_xticks([])
        axes[row, 0].set_yticks([0, 0.5, 1])
    for col in range(len(Tplot)):
        if col > 0:
            axes[0, col].set_yticks([])
            axes[1, col].set_yticks([])
            axes[2, col].set_yticks([])
    axes[2, 0].set_xticks([0, 0.5, 1])

    plt.show()

def plot_populations(Tdata, Ydata, Tplot):
    """Contour plots of P1, P2 at selected time indices in `Tplot`"""

    P1 = np.empty((len(Tplot), Ny, Nx))
    P2 = np.empty((len(Tplot), Ny, Nx))

    for i, ti in enumerate(Tplot):
        P1[i, :, :] = np.reshape(Ydata[0 * Nx * Ny:1 * Nx * Ny, ti], (Ny, Nx))
        P2[i, :, :] = np.reshape(Ydata[1 * Nx * Ny:2 * Nx * Ny, ti], (Ny, Nx))

    fig, axes = plt.subplots(2, len(Tplot), figsize=(10, 6), constrained_layout=True)

    vminP = min(P1.min(), P2.min())
    vmaxP = max(P1.max(), P2.max())
    normP = colors.Normalize(vmin=vminP, vmax=vmaxP)

    for j in range(len(Tplot)):
        im1 = axes[0, j].contourf(x, y, P1[j, :, :].T, levels=50, cmap=cmap_u1, norm=normP)#, extend="min")
        im2 = axes[1, j].contourf(x, y, P2[j, :, :].T, levels=50, cmap=cmap_u2, norm=normP)#, extend="min")
        axes[0, j].set_title(f"t = {Tdata[Tplot[j]]:.0f}")

    # Colourbars
    cbar1 = fig.colorbar(im1, ax=axes[0, :], location="right")
    cbar2 = fig.colorbar(im2, ax=axes[1, :], location="right")
    for cb, label in zip((cbar1, cbar2), ("$u_1$", "$u_2$")):
        cb.formatter = ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        cb.locator = MaxNLocator(nbins=4)
        cb.update_ticks()
        cb.set_label(label, rotation=0, labelpad=30)
        cb.ax.yaxis.set_label_position('left')

    # Axis labels
    axes[0, 0].set_ylabel("$y$", rotation=0)
    axes[1, 0].set_ylabel("$y$", rotation=0)
    axes[1, 0].set_xlabel("$x$")
    for col in range(len(Tplot)):
        axes[1, col].set_xlabel("$x$")

    # Simplify ticks
    for row in range(2):
        for col in range(len(Tplot)):
            if row == 0:
                axes[row, col].set_xticks([])
        axes[row, 0].set_yticks([0, 0.5, 1])
    for col in range(1, len(Tplot)):
        axes[0, col].set_yticks([])
        axes[1, col].set_yticks([])
    axes[1, 0].set_xticks([0, 0.5, 1])

    plt.show()

# Execute plotting functions
plot_populations(Tdata, Ydata, Tplot)
plot_everything(Tdata, Ydata, Tplot)
plot_total_mass(Tdata, Ydata, dx)
