##############################################################################
# 2D diffusion–advection–reaction model for a multi-species bacterial system
# with competition for space (pressure) and a shared nutrient.
#
# State:
#   P_i(x,t): bacterial density for species i = 1,2,3
#   C(x,t)  : substrate (nutrient) density
#
# Movement:
#   Drift down the pressure gradient (negative chemotaxis), with pressure
#   Pr = P1 + P2 + P3. Diffusion for both bacteria and substrate.
#
# Reactions:
#   Birth/death for P_i; substrate consumption G(P1,P2,P3,C).
#
# Domain:
#   2D rectangle [0,Lx] x [0,Ly] with no-flux boundary conditions.
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

# Wall-clock timer
start_time = time.time()

################################################################
# -------------------------- PARAMETERS ----------------------- #
################################################################

global N_space, N_y, Lx, Ly, dx, dy, x, y, x_mp, D, A, d, K1, K2, gam, beta, sigma

np.random.seed(42)

# Spatial grid (for generating random microcolonies for ICs)
nx, ny = 100, 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Circular support (main colony footprint) used to initialise components
main_radius = 0.2
center = (0.5, 0.5)
distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
support = distance_from_center <= main_radius

# Three components (species) initial pattern: many small circles randomly placed
total_density = 1.0
components = np.zeros((3, nx, ny))
num_circles = 300
small_radius = 0.05

for _ in range(num_circles):
    # Random centre inside main circle
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, main_radius - small_radius)
    cx, cy = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
    # Disc mask
    small_circle = (X - cx)**2 + (Y - cy)**2 <= small_radius**2
    # Assign disc to a random species
    component_index = np.random.choice(3)
    components[component_index] += small_circle.astype(float)

# Mask outside support; normalise so summed components equal total_density
for i in range(3):
    components[i][~support] = 0
density_sum = np.sum(components, axis=0)
density_sum[density_sum == 0] = 1
components = components / density_sum * total_density

# Solver grid (used by the PDE solver)
N_space = 100
N_y = 100
Lx = 1
Ly = 1

# Small tolerance to avoid division by zero
eps = 10**(-9)

# Grid steps and solver coordinates
dx = Lx/(N_space - 1)
dy = Ly/(N_y - 1)
y = dy*np.arange(0, N_y, 1)
x = dx*np.arange(0, N_space, 1)

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

# Helper IC builders (Gaussian bump / smooth bump on a disc)
def bump_function(x, y, xm, ym, rcut):
    r = np.sqrt((x - xm)**2 + (y - ym)**2)
    inside = r < rcut
    result = np.zeros_like(r)
    result[inside] = np.exp(-1 / (1 - r[inside]**2))
    return result

def initial_conditions(shape, center, bump_width, background_mass):
    """
    Gaussian-like bump plus uniform background for ICs.
    """
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    squared_distance = (xx - center[0])**2 + (yy - center[1])**2
    bump = np.exp(-squared_distance / (2 * bump_width**2))
    initial_conditions = bump + background_mass
    return initial_conditions

# IC convenience parameters (unused in final IC choice below)
grid_shape = (N_space, N_y)
bump_width = 5
background_mass = 0

# Initial conditions for P1, P2, P3 from the random microcolony generator above
P10 = components[0]
P20 = components[1]
P30 = components[2]

# Initial substrate C (uniform)
C0 = 5*np.ones([N_space, N_y])

# Pressure initialisation (filled later in rhs)
Pr = np.zeros([N_space, N_y])

################################################################
# ----------------------- RIGHT-HAND SIDE --------------------- #
################################################################
def rhsode(t, y):
    """
    Semi-discrete RHS for the PDE system (method of lines).
    y is a flattened vector containing P1, P2, P3, C on the solver grid.
    """

    # Flux tensors for P1,P2,P3 and C
    fP1 = np.zeros([N_space, N_y, 4])
    fP2 = np.zeros([N_space, N_y, 4])
    fP3 = np.zeros([N_space, N_y, 4])
    fC  = np.zeros([N_space, N_y, 4])

    # Unpack state
    P1 = np.reshape(y[0:N_space*N_y], ([N_space, N_y]))
    P2 = np.reshape(y[N_space*N_y:2*N_space*N_y], ([N_space, N_y]))
    P3 = np.reshape(y[2*N_space*N_y:3*N_space*N_y], ([N_space, N_y]))
    C  = np.reshape(y[3*N_space*N_y:4*N_space*N_y], ([N_space, N_y]))

    # Pressure
    Pr = alpha*(P1 + P2 + P3)

    # Substrate fluxes (central differences with simple boundary fluxes)
    fC[0:N_space-1, :, 0] = - d * np.diff(C, axis=0)
    fC[1:N_space, :, 1]   = fC[0:N_space-1, :, 0]
    fC[:, 0:N_y-1,  2]    = - d * np.diff(C, axis=1)
    fC[:, 1:N_y,    3]    = fC[:, 0:N_y-1, 2]

    # Boundary contributions (in/outflow proxies)
    fC[N_space-1, :, 0] = -Cinf * np.ones(N_y)   # right
    fC[0,        :, 1] =  Cinf * np.ones(N_y)    # left
    fC[:, N_y-1,  2] = -Cinf * np.ones(N_space)  # top
    fC[:, 0,      3] =  Cinf * np.ones(N_space)  # bottom

    # Prepare arrays for time derivatives
    dP1dt = np.zeros([N_space, N_y])
    dP2dt = np.zeros([N_space, N_y])
    dP3dt = np.zeros([N_space, N_y])
    dCdt  = np.zeros([N_space, N_y])

    # Upwind-like split of pressure gradients (for drift direction)
    DPrxp = np.where(np.diff(Pr, axis=0) > 0, np.diff(Pr, axis=0), 0)
    DPrxm = np.where(np.diff(Pr, axis=0) < 0, np.diff(Pr, axis=0), 0)
    DPryp = np.where(np.diff(Pr, axis=1) > 0, np.diff(Pr, axis=1), 0)
    DPrym = np.where(np.diff(Pr, axis=1) < 0, np.diff(Pr, axis=1), 0)

    # Bacterial fluxes (diffusion + drift down ∇Pr)
    fP1[0:N_space-1, :, 0] = - D1 * np.diff(P1, axis=0) - A1 * (DPrxp * P1[0:N_space-1, :] + DPrxm * P1[1:N_space, :])
    fP1[1:N_space,   :, 1] = fP1[0:N_space-1, :, 0]
    fP1[:, 0:N_y-1,  2]    = - D1 * np.diff(P1, axis=1) - A1 * (DPryp * P1[:, 0:N_y-1] + DPrym * P1[:, 1:N_y])
    fP1[:, 1:N_y,    3]    = fP1[:, 0:N_y-1, 2]

    fP2[0:N_space-1, :, 0] = - D2 * np.diff(P2, axis=0) - A2 * (DPrxp * P2[0:N_space-1, :] + DPrxm * P2[1:N_space, :])
    fP2[1:N_space,   :, 1] = fP2[0:N_space-1, :, 0]
    fP2[:, 0:N_y-1,  2]    = - D2 * np.diff(P2, axis=1) - A2 * (DPryp * P2[:, 0:N_y-1] + DPrym * P2[:, 1:N_y])
    fP2[:, 1:N_y,    3]    = fP2[:, 0:N_y-1, 2]

    fP3[0:N_space-1, :, 0] = - D3 * np.diff(P3, axis=0) - A3 * (DPrxp * P3[0:N_space-1, :] + DPrxm * P3[1:N_space, :])
    fP3[1:N_space,   :, 1] = fP3[0:N_space-1, :, 0]
    fP3[:, 0:N_y-1,  2]    = - D3 * np.diff(P3, axis=1) - A3 * (DPryp * P3[:, 0:N_y-1] + DPrym * P3[:, 1:N_y])
    fP3[:, 1:N_y,    3]    = fP3[:, 0:N_y-1, 2]

    # Divergence of fluxes → transport contribution
    dP1dt = 1/(dx**2) * (fP1[:, :, 1] - fP1[:, :, 0]) + 1/(dy**2) * (fP1[:, :, 3] - fP1[:, :, 2])
    dP2dt = 1/(dx**2) * (fP2[:, :, 1] - fP2[:, :, 0]) + 1/(dy**2) * (fP2[:, :, 3] - fP2[:, :, 2])
    dP3dt = 1/(dx**2) * (fP3[:, :, 1] - fP3[:, :, 0]) + 1/(dy**2) * (fP3[:, :, 3] - fP3[:, :, 2])
    dCdt  = 1/(dx**2) * (fC[:,  :, 1] - fC[:,  :, 0]) + 1/(dy**2)  * (fC[:,  :, 3] - fC[:,  :, 2])

    # Simple check for numerical mass balance of transport
    if abs(sum(sum(dP1dt))) > 10**(-6):
        print('The flux is not 0! ' + str(sum(sum(dP1dt))))

    # Substrate kinetics (Monod-type uptake by each species; signs chosen as consumption)
    kinetics_lig = - ( r1/sig1 * P1 * C/(K1 + C + eps)
                     + r2/sig2 * P2 * C/(K2 + C + eps)
                     + r3/sig3 * P3 * C/(K2 + C + eps) )
    dCdt = dCdt + kinetics_lig

    # Population kinetics (growth with Monod uptake, quadratic self-limitation)
    dP1dt += P1 * (r1 * C/(K1 + C) - b1 * Pr)
    dP2dt += P2 * (r2 * C/(K2 + C) - b2 * Pr)
    dP3dt += P3 * (r3 * C/(K3 + C) - b3 * Pr)

    # Pack RHS back to a vector
    dudt = np.zeros(4 * N_space * N_y)
    dudt[0:N_space*N_y]                 = dP1dt.flatten()
    dudt[N_space*N_y:2*N_space*N_y]     = dP2dt.flatten()
    dudt[2*N_space*N_y:3*N_space*N_y]   = dP3dt.flatten()
    dudt[3*N_space*N_y:4*N_space*N_y]   = dCdt.flatten()
    return dudt

################################################################
# ----------------------- INITIAL STATE ----------------------- #
################################################################

# Stack initial fields into a single vector U0 = [P1, P2, P3, C]
U0 = np.zeros(4 * N_space * N_y)
U0[0:N_space*N_y]                 = P10.flatten()
U0[N_space*N_y:2*N_space*N_y]     = P20.flatten()
U0[2*N_space*N_y:3*N_space*N_y]   = P30.flatten()
U0[3*N_space*N_y:4*N_space*N_y]   = C0.flatten()

################################################################
# -------------------------- SOLVE ---------------------------- #
################################################################

Tend = 200#500
interval = [0, Tend]
tsave = np.arange(0, Tend + 1, 1)

solution = solve_ivp(rhsode, interval, U0, method='RK45', t_eval=tsave)

# Optional CSV saving (left commented out intentionally)
# np.savetxt("test_y.csv", solution.y, delimiter=',')
# np.savetxt("test_t.csv", solution.t, delimiter=',')

print('Duration: ' + str(time.time() - start_time))

################################################################
# -------------------------- PLOTTING ------------------------- #
################################################################

# Times to plot (indices into tsave)
Tplot = np.arange(0, len(tsave), 1)
Tplot = [0,25,50]#[0,100,200]

# Extract time series arrays from solver
Tdata = solution.t
Ydata = solution.y

# (Optional) global plot styling (kept commented)
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

P1 = np.empty((len(Tplot),N_y,N_space))
P2 = np.empty((len(Tplot),N_y,N_space))
P3 =np.empty((len(Tplot),N_y,N_space))
C = np.empty((len(Tplot),N_y,N_space))




fig, axes = plt.subplots(4, len(Tplot), figsize=(10, 10), constrained_layout = True)

for t in range(len(Tplot)):
    
    P1[t,:,:] = np.reshape(Ydata[0:N_space*N_y,                 Tplot[t]], (N_y, N_space))
    P2[t,:,:] = np.reshape(Ydata[N_space*N_y:2*N_space*N_y,     Tplot[t]], (N_y, N_space))
    P3[t,:,:] = np.reshape(Ydata[2*N_space*N_y:3*N_space*N_y,   Tplot[t]], (N_y, N_space))
    C[t,:,:]  = np.reshape(Ydata[3*N_space*N_y:4*N_space*N_y,   Tplot[t]], (N_y, N_space))


vmax = max(P1.max(),P2.max(),P3.max())

for t in range(len(Tplot)):

# Substrate (independent colour scale)
    im1 = axes[0, t].contourf(x, y, C[t,:,:].T, cmap='Blues', levels=50)
    #im1 = axes[0, 1].contourf(x, y, C[1,:,:].T, cmap='Greens', levels=50)
    #im1 = axes[0, 2].contourf(x, y, C[2,:,:].T, cmap='Greens', levels=50)
    im2 = axes[1,t].contourf(x, y, P1[t,:,:].T, cmap='Greens', levels=50, vmin=0, vmax = vmax)
    im3 = axes[2,t].contourf(x, y, P2[t,:,:].T, cmap='Oranges', levels=50, vmin=0, vmax = vmax)
    im4 = axes[3,t].contourf(x, y, P3[t,:,:].T, cmap='Purples', levels=50, vmin=0, vmax = vmax)
    
    axes[0,t].set_title(f"t = {Tplot[t]:.2f}")
    
#cax = fig.add_axes([0.95, 0.9, 0.02, 0.2])  # [left, bottom, width, height]


cbar1 = fig.colorbar(im1, ax = axes[0,:], location = 'right')
cbar1.formatter = ScalarFormatter(useMathText=True)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar1.locator = MaxNLocator(nbins=4)
cbar1.update_ticks()
axes[0, 0].set_ylabel("Substrate c")


cbar2 = fig.colorbar(im2, ax=axes[1,:], location = 'right')
cbar3 = fig.colorbar(im3, ax=axes[2,:], location = 'right')
cbar4 = fig.colorbar(im4, ax=axes[3,:], location = 'right')
cbar2.formatter = ScalarFormatter(useMathText=True)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar2.locator = MaxNLocator(nbins=4)
cbar2.update_ticks()
cbar3.formatter = ScalarFormatter(useMathText=True)
cbar3.formatter.set_powerlimits((0, 0))
cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar3.locator = MaxNLocator(nbins=4)
cbar3.update_ticks()
cbar4.formatter = ScalarFormatter(useMathText=True)
cbar4.formatter.set_powerlimits((0, 0))
cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar4.locator = MaxNLocator(nbins=4)
cbar4.update_ticks()
axes[1, 0].set_ylabel("Population $u_1$")
axes[2, 0].set_ylabel("Population $u_2$")
axes[3, 0].set_ylabel("Population $u_3$")

#axes[:,0:1].set_xticks([])

for i in (0,1,2):
    axes[i,0].set_xticks([])
    axes[i,1].set_xticks([])
    axes[i,2].set_xticks([])
    
for i in (1,2):
    axes[0,i].set_yticks([])
    axes[1,i].set_yticks([])
    axes[2,i].set_yticks([])
    axes[3,i].set_yticks([])



plt.show()


def format_cbar(cbar):
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

# Make one figure per time index (column), each with 3 rows: u1, u2, u3
for it, col in enumerate(Tplot):
    tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"

    # Per-figure (per-time) colour scaling; switch to global scaling if preferred
    normP1 = colors.Normalize(vmin=np.min(P1[it]), vmax=np.max(P1[it]))
    normP2 = colors.Normalize(vmin=np.min(P2[it]), vmax=np.max(P2[it]))
    normP3 = colors.Normalize(vmin=np.min(P3[it]), vmax=np.max(P3[it]))

    figP, axesP = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=True, sharex=True)

    imsP = [
        axesP[0].contourf(x, y, P1[it].T, levels=50, cmap='Greens', norm=normP1),
        axesP[1].contourf(x, y, P2[it].T, levels=50, cmap='Oranges', norm=normP2),
        axesP[2].contourf(x, y, P3[it].T, levels=50, cmap='Purples', norm=normP3),
    ]

    axesP[0].set_title(tlabel)
    axesP[2].set_xlabel("x")
    axesP[0].set_ylabel(r"$u_1$")
    axesP[1].set_ylabel(r"$u_2$")
    axesP[2].set_ylabel(r"$u_3$")

    # Tidy ticks if domain is [0,1] × [0,1]
    try:
        axesP[2].set_xticks([0, 0.5, 1])
        for ax in axesP:
            ax.set_yticks([0, 0.5, 1])
    except Exception:
        pass

    # One colourbar per subplot
    for ax, im in zip(axesP, imsP):
        cb = figP.colorbar(im, ax=ax, location='right')
        format_cbar(cb)

# Render all figures created in the loop
plt.show()



from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Adjustable settings ---
density_threshold = 0.1   # Mask regions where densities are below this value
use_global_scaling = False  # Set True to use the same colour scale for all times

def format_cbar(cbar):
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

# --- Optional global normalisation ---
if use_global_scaling:
    P1_all = np.array([np.max(p) for p in P1])
    normP1_global = colors.Normalize(vmin=0.0, vmax=float(np.max(P1_all)) or 1.0)
else:
    normP1_global = None

# --- Plotting loop ---
for it, col in enumerate(Tplot):
    tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"

    # Choose scaling
    if use_global_scaling and normP1_global:
        normP1 = normP1_global
    else:
        normP1 = colors.Normalize(vmin=np.min(P1[it]), vmax=np.max(P1[it]) or 1.0)

    normP2 = colors.Normalize(vmin=np.min(P2[it]), vmax=np.max(P2[it]) or 1.0)
    normP3 = colors.Normalize(vmin=np.min(P3[it]), vmax=np.max(P3[it]) or 1.0)

    # Mask low-density regions
    P1_masked = np.ma.masked_where(P1[it].T <= density_threshold, P1[it].T)
    P2_masked = np.ma.masked_where(P2[it].T <= density_threshold, P2[it].T)
    P3_masked = np.ma.masked_where(P3[it].T <= density_threshold, P3[it].T)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5), constrained_layout=True)

    # Overlay all three populations
    im1 = ax.contourf(x, y, P1_masked, levels=5, cmap='Greens',  norm=normP1, alpha=0.9, zorder=1)
    im2 = ax.contourf(x, y, P2_masked, levels=5, cmap='Oranges', norm=normP2, alpha=0.9, zorder=2)
    im3 = ax.contourf(x, y, P3_masked, levels=5, cmap='Purples', norm=normP3, alpha=0.9, zorder=3)

    # Labels and title
    ax.set_title(tlabel)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Neat ticks
    try:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    except Exception:
        pass

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#2ca02c', label=r"$u_1$"),
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#ff7f0e', label=r"$u_2$"),
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#9467bd', label=r"$u_3$"),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True)

    # --- Single colourbar (for P1 only) ---
    cb = fig.colorbar(im1, ax=ax, fraction=0.04, pad=0.04)
    #cb.set_label(r"$u_1$ density")
    format_cbar(cb)


# Show all figures
plt.show()
