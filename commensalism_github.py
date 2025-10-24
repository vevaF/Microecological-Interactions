##############################################################################
# 2D diffusion–advection–reaction model (COMMENSALISM scenario)
#
# Three bacterial species P1, P2, P3 move by diffusion and drift down the
# population pressure gradient (negative chemotaxis), where:
#   Pr = P1 + P2 + P3  (scaled by alpha).
#
# Substrate chain (commensalism):
#   - C1 is consumed by P1 and (partly) converted into C2.
#   - C2 is consumed by P2 and (partly) converted into C3.
#   - C3 is consumed by P3.
#
# PDE structure:
#   ∂Pi/∂t = ∇·(Di ∇Pi - Ai Pi ∇Pr) + Fi(Pi, Ci)
#   ∂Cj/∂t = d ΔCj + Gj(...)  (with production/consumption as above)
#
# Domain: [0,Lx] × [0,Ly], no-flux BCs for bacteria; simple inflow/outflow-like
# terms used for substrates via C?inf.
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
# -------------------- INITIAL CONDITION BUILDER ---------------#
################################################################
# Generate random microcolonies within a circular support to define
# components[0], components[1], components[2] used for P1, P2, P3 ICs.

np.random.seed(42)

# IC grid (for shaping microcolonies; matches solver grid below: 100×100 on [0,1]²)
nx, ny = 100, 100
x_ic = np.linspace(0, 1, nx)
y_ic = np.linspace(0, 1, ny)
X_ic, Y_ic = np.meshgrid(x_ic, y_ic)

# Circular support (main colony footprint)
main_radius = 0.2
center = (0.5, 0.5)
dist = np.sqrt((X_ic - center[0])**2 + (Y_ic - center[1])**2)
support = dist <= main_radius

# Randomly sprinkle many small discs among the three species
components = np.zeros((3, nx, ny))
total_density = 1.0
num_circles = 300
small_radius = 0.05

for _ in range(num_circles):
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, main_radius - small_radius)
    cx = center[0] + r*np.cos(theta)
    cy = center[1] + r*np.sin(theta)
    disc = (X_ic - cx)**2 + (Y_ic - cy)**2 <= small_radius**2
    idx = np.random.choice(3)
    components[idx] += disc.astype(float)

# Mask outside the support and normalise to total_density
for i in range(3):
    components[i][~support] = 0
den = np.sum(components, axis=0)
den[den == 0] = 1
components = components / den * total_density

################################################################
# -------------------------- PARAMETERS ----------------------- #
################################################################

global N_space, N_y, Lx, Ly, dx, dy, x, y, x_mp, D, A, d, K1, K2, gam, beta, sigma

# Solver grid
N_space = 100
N_y = 100
Lx = 1
Ly = 1
eps = 10**(-9)

dx = Lx/(N_space - 1)
dy = Ly/(N_y - 1)
y = dy*np.arange(0, N_y, 1)
x = dx*np.arange(0, N_space, 1)

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
r1 = 1#0.5
r2 = 1#0.5
r3 = 1#0.5
b1 = 0.1#0.75
b2 = 0.1#0.75
b3 = 0.1#0.75
sig = 10     # overall yield / stoichiometric scaling (used below)


# Substrate boundary “inflow/outflow” strengths
C1inf = 0
C2inf = 0
C3inf = 0

# Optional IC helpers (kept for reference)
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
    x = np.arange(shape[0]); y = np.arange(shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    sq = (xx - center[0])**2 + (yy - center[1])**2
    bump = np.exp(-sq / (2 * bump_width**2))
    return bump + background_mass

grid_shape = (N_space, N_y)
bump_width = 5
background_mass = 0

# Initial conditions for populations (from components above)
# Note: order intentionally matches your code: P1<-components[1], P2<-components[0], P3<-components[2]
P10 = components[1]
P20 = components[0]
P30 = components[2]

# Initial substrates: C1 present, C2 and C3 nearly zero
C10 = 5*np.ones([N_space, N_y])
C20 = np.zeros([N_space, N_y])#eps*np.ones([N_space, N_y])
C30 = np.zeros([N_space, N_y])#eps*np.ones([N_space, N_y])

# Pressure (filled in RHS)
Pr = np.zeros([N_space, N_y])

################################################################
# ----------------------- RIGHT-HAND SIDE --------------------- #
################################################################
def rhsode(t, y):
    """
    Semi-discrete RHS for the commensalism PDE system (method of lines).
    State vector y stacks P1, P2, P3, C1, C2, C3 on the (N_space × N_y) grid.
    """

    n = 6  # number of fields

    # Flux containers
    fP1 = np.zeros([N_space, N_y, n])
    fP2 = np.zeros([N_space, N_y, n])
    fP3 = np.zeros([N_space, N_y, n])
    fC1 = np.zeros([N_space, N_y, n])
    fC2 = np.zeros([N_space, N_y, n])
    fC3 = np.zeros([N_space, N_y, n])

    # Unpack state
    P1 = np.reshape(y[0:N_space*N_y],                ([N_space, N_y]))
    P2 = np.reshape(y[N_space*N_y:2*N_space*N_y],    ([N_space, N_y]))
    P3 = np.reshape(y[2*N_space*N_y:3*N_space*N_y],  ([N_space, N_y]))
    C1 = np.reshape(y[3*N_space*N_y:4*N_space*N_y],  ([N_space, N_y]))
    C2 = np.reshape(y[4*N_space*N_y:5*N_space*N_y],  ([N_space, N_y]))
    C3 = np.reshape(y[5*N_space*N_y:6*N_space*N_y],  ([N_space, N_y]))

    # Pressure
    Pr = alpha*(P1 + P2 + P3)

    # Substrate fluxes (central differences + simple boundary contributions)
    # C1
    fC1[0:N_space-1, :, 0] = - d * np.diff(C1, axis=0)
    fC1[1:N_space,   :, 1] = fC1[0:N_space-1, :, 0]
    fC1[:, 0:N_y-1,  2]    = - d * np.diff(C1, axis=1)
    fC1[:, 1:N_y,    3]    = fC1[:, 0:N_y-1, 2]
    fC1[N_space-1, :, 0] = -C1inf * np.ones(N_y)
    fC1[0,        :, 1] =  C1inf * np.ones(N_y)
    fC1[:, N_y-1,  2] = -C1inf * np.ones(N_space)
    fC1[:, 0,      3] =  C1inf * np.ones(N_space)

    # C2
    fC2[0:N_space-1, :, 0] = - d * np.diff(C2, axis=0)
    fC2[1:N_space,   :, 1] = fC2[0:N_space-1, :, 0]
    fC2[:, 0:N_y-1,  2]    = - d * np.diff(C2, axis=1)
    fC2[:, 1:N_y,    3]    = fC2[:, 0:N_y-1, 2]
    fC2[N_space-1, :, 0] = -C2inf * np.ones(N_y)
    fC2[0,        :, 1] =  C2inf * np.ones(N_y)
    fC2[:, N_y-1,  2] = -C2inf * np.ones(N_space)
    fC2[:, 0,      3] =  C2inf * np.ones(N_space)

    # C3
    fC3[0:N_space-1, :, 0] = - d * np.diff(C3, axis=0)
    fC3[1:N_space,   :, 1] = fC3[0:N_space-1, :, 0]
    fC3[:, 0:N_y-1,  2]    = - d * np.diff(C3, axis=1)
    fC3[:, 1:N_y,    3]    = fC3[:, 0:N_y-1, 2]
    fC3[N_space-1, :, 0] = -C3inf * np.ones(N_y)
    fC3[0,        :, 1] =  C3inf * np.ones(N_y)
    fC3[:, N_y-1,  2] = -C3inf * np.ones(N_space)
    fC3[:, 0,      3] =  C3inf * np.ones(N_space)

    # Allocate time-derivative arrays
    dP1dt = np.zeros([N_space, N_y])
    dP2dt = np.zeros([N_space, N_y])
    dP3dt = np.zeros([N_space, N_y])
    dC1dt = np.zeros([N_space, N_y])
    dC2dt = np.zeros([N_space, N_y])
    dC3dt = np.zeros([N_space, N_y])

    # Upwind-like split on ∇Pr for drift
    DPrxp = np.where(np.diff(Pr, axis=0) > 0, np.diff(Pr, axis=0), 0)
    DPrxm = np.where(np.diff(Pr, axis=0) < 0, np.diff(Pr, axis=0), 0)
    DPryp = np.where(np.diff(Pr, axis=1) > 0, np.diff(Pr, axis=1), 0)
    DPrym = np.where(np.diff(Pr, axis=1) < 0, np.diff(Pr, axis=1), 0)

    # Bacterial fluxes: diffusion + drift down ∇Pr
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
    kinetics_C2 =  sig/2 * r1 * P1 * C1/ (K1 + C1 + eps) - sig * r2 * P2 * C2 / (K2 + C2 + eps)
    kinetics_C3 =  sig/2 * r2 * P2 * C2/ (K2 + C2 + eps) - sig * r3 * P3 * C3 / (K3 + C3 + eps)

    dC1dt += kinetics_C1
    dC2dt += kinetics_C2
    dC3dt += kinetics_C3

    # Population kinetics: growth (Monod in Ci) and quadratic self-limitation
    dP1dt +=  P1 * (r1 * C1 / (K1 + C1) - b1 * Pr)
    dP2dt +=  P2 * (r2 * C2 / (K2 + C2) - b2 * Pr)
    dP3dt +=  P3 * (r3 * C3 / (K3 + C3) - b3 * Pr)

    # Pack back to vector
    dudt = np.zeros(n * N_space * N_y)
    dudt[0:N_space*N_y]                 = dP1dt.flatten()
    dudt[N_space*N_y:2*N_space*N_y]     = dP2dt.flatten()
    dudt[2*N_space*N_y:3*N_space*N_y]   = dP3dt.flatten()
    dudt[3*N_space*N_y:4*N_space*N_y]   = dC1dt.flatten()
    dudt[4*N_space*N_y:5*N_space*N_y]   = dC2dt.flatten()
    dudt[5*N_space*N_y:6*N_space*N_y]   = dC3dt.flatten()
    return dudt

################################################################
# ----------------------- INITIAL STATE ----------------------- #
################################################################

U0 = np.zeros(6 * N_space * N_y)
U0[0:N_space*N_y]                 = P10.flatten()
U0[N_space*N_y:2*N_space*N_y]     = P20.flatten()
U0[2*N_space*N_y:3*N_space*N_y]   = P30.flatten()
U0[3*N_space*N_y:4*N_space*N_y]   = C10.flatten()
U0[4*N_space*N_y:5*N_space*N_y]   = C20.flatten()
U0[5*N_space*N_y:6*N_space*N_y]   = C30.flatten()

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

print('Duration: ' + str(time.time() - start_time))

################################################################
# -------------------------- PLOTTING ------------------------- #
################################################################

# Tplot = np.arange(0, len(tsave), 1)
# Tdata = solution.t
# Ydata = solution.y

# plt.rcParams.update({
#     'font.size': 16,
#     'axes.titlesize': 20,
#     'axes.labelsize': 18,
#     'xtick.labelsize': 16,
#     'ytick.labelsize': 16,
#     'legend.fontsize': 16,
#     'figure.titlesize': 22
# })

# for t in range(len(Tplot)):
#     P1 = np.reshape(Ydata[0:N_space*N_y,               Tplot[t]], ([N_y, N_space]))
#     P2 = np.reshape(Ydata[N_space*N_y:2*N_space*N_y,   Tplot[t]], ([N_y, N_space]))
#     P3 = np.reshape(Ydata[2*N_space*N_y:3*N_space*N_y, Tplot[t]], ([N_y, N_space]))
#     C1 = np.reshape(Ydata[3*N_space*N_y:4*N_space*N_y, Tplot[t]], ([N_y, N_space]))
#     C2 = np.reshape(Ydata[4*N_space*N_y:5*N_space*N_y, Tplot[t]], ([N_y, N_space]))
#     C3 = np.reshape(Ydata[5*N_space*N_y:6*N_space*N_y, Tplot[t]], ([N_y, N_space]))

#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#     # Substrates
#     im1 = axes[0, 0].contourf(x, y, C1.transpose()); axes[0, 0].set_title("C1"); fig.colorbar(im1, ax=axes[0, 0])
#     im2 = axes[0, 1].contourf(x, y, C2.transpose()); axes[0, 1].set_title("C2"); fig.colorbar(im2, ax=axes[0, 1])
#     im3 = axes[0, 2].contourf(x, y, C3.transpose()); axes[0, 2].set_title("C3"); fig.colorbar(im3, ax=axes[0, 2])

#     # Populations (as in your original plotting logic)
#     im4 = axes[1, 0].contourf(x, y, P1.transpose() - P2.transpose(), vmin=0); axes[1, 0].set_title("P1"); fig.colorbar(im4, ax=axes[1, 0])
#     im5 = axes[1, 1].contourf(x, y, P2.transpose() - P3.transpose(), vmin=0); axes[1, 1].set_title("P2"); fig.colorbar(im5, ax=axes[1, 1])
#     im6 = axes[1, 2].contourf(x, y, P3.transpose(), vmin=0);                 axes[1, 2].set_title("P3"); fig.colorbar(im6, ax=axes[1, 2])

#     fig.suptitle("Contour Plots", fontsize=16)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     filename = f"commensalism_{t}.png"
#     #uncomment to save plots
#     #plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.show()




Tplot = [0,2,5]

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


# Three substrates (replace former P1/P2/P3)
P1 = np.empty((len(Tplot), N_y, N_space))
P2 = np.empty((len(Tplot), N_y, N_space))
P3 = np.empty((len(Tplot), N_y, N_space))


fig, axes = plt.subplots(3, len(Tplot), figsize=(10, 8), constrained_layout = True)

for t in range(len(Tplot)):
    # Substrates now occupy the first three blocks
    P1[t,:,:] = np.reshape(Ydata[0*N_space*N_y:1*N_space*N_y, Tplot[t]], (N_y, N_space))
    P2[t,:,:] = np.reshape(Ydata[1*N_space*N_y:2*N_space*N_y, Tplot[t]], (N_y, N_space))
    P3[t,:,:] = np.reshape(Ydata[2*N_space*N_y:3*N_space*N_y, Tplot[t]], (N_y, N_space))

#P1[0,:,:] = 0*np.ones([N_y, N_space])
#P1[1,:,:] = 5*np.ones([N_y, N_space])
#P1[2,:,:] = 5*np.ones([N_y, N_space])

normP1 = colors.Normalize(vmin=np.min(P1), vmax=np.max(P1))
normP2 = colors.Normalize(vmin=np.min(P2), vmax=np.max(P2))
normP3 = colors.Normalize(vmin=np.min(P3), vmax=np.max(P3))

# Independent colour scales for substrates (as with original substrate row)
for t in range(len(Tplot)):
    im1 = axes[0, t].contourf(x, y, P1[t,:,:].T, cmap='Greens', vmin=np.min(P1), vmax = np.max(P1), levels=50)
    im2 = axes[1, t].contourf(x, y, P2[t,:,:].T, cmap='Oranges', vmin=np.min(P2), vmax = np.max(P2), levels=50)
    im3 = axes[2, t].contourf(x, y, P3[t,:,:].T, cmap='Purples', vmin=np.min(P3), vmax = np.max(P3), levels=50)

    axes[0, t].set_title(f"t = {Tplot[t]:.2f}")

# Colourbars (same formatting as your original)
cbar1 = fig.colorbar(im1, ax = axes[0,:], location = 'right')
cbar1.formatter = ScalarFormatter(useMathText=True)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar1.locator = MaxNLocator(nbins=4)
cbar1.update_ticks()
axes[0, 0].set_ylabel("Population $u_1$")

cbar2 = fig.colorbar(im2, ax=axes[1,:], norm=normP2, location = 'right')
cbar3 = fig.colorbar(im3, ax=axes[2,:], norm=normP3, location = 'right')
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
axes[1, 0].set_ylabel("Population $u_2$")
axes[2, 0].set_ylabel("Population $u_3$")

# Tick visibility (same logic, adapted to 3 rows)
for i in (0,1,2):
    axes[i,0].set_xticks([])
    axes[i,1].set_xticks([])
    axes[i,2].set_xticks([])
    axes[i,0].set_yticks([0,0.5,1])

for i in (1,2):
    axes[0,i].set_yticks([])
    axes[1,i].set_yticks([])
    axes[2,i].set_yticks([])
    axes[2,i].set_xticks([0,0.5,1])
axes[2,0].set_xticks([0,0.5,1])

# Three substrates (replace former P1/P2/P3)
C1 = np.empty((len(Tplot), N_y, N_space))
C2 = np.empty((len(Tplot), N_y, N_space))
C3 = np.empty((len(Tplot), N_y, N_space))


fig, axes = plt.subplots(3, len(Tplot), figsize=(10, 8), constrained_layout = True)

for t in range(len(Tplot)):
    # Substrates now occupy the first three blocks
    C1[t,:,:] = np.reshape(Ydata[3*N_space*N_y:4*N_space*N_y, Tplot[t]], (N_y, N_space))
    C2[t,:,:] = np.reshape(Ydata[4*N_space*N_y:5*N_space*N_y, Tplot[t]], (N_y, N_space))
    C3[t,:,:] = np.reshape(Ydata[5*N_space*N_y:6*N_space*N_y, Tplot[t]], (N_y, N_space))

normC1 = colors.Normalize(vmin=np.min(C1), vmax=np.max(C1))
normC2 = colors.Normalize(vmin=np.min(C2), vmax=np.max(C2))
normC3 = colors.Normalize(vmin=np.min(C3), vmax=np.max(C3))
print(np.min(C3))

# Independent colour scales for substrates (as with original substrate row)
for t in range(len(Tplot)):
    im1 = axes[0, t].contourf(x, y, C1[t,:,:].T, cmap='Blues', norm=normC1, levels=50)
    im2 = axes[1, t].contourf(x, y, C2[t,:,:].T, cmap='Blues', norm=normC2, levels=50)
    im3 = axes[2, t].contourf(x, y, C3[t,:,:].T, cmap='Blues', norm=normC3, levels=50)

    axes[0, t].set_title(f"t = {Tplot[t]:.2f}")

# Colourbars (same formatting as your original)
cbar1 = fig.colorbar(im1, ax = axes[0,:], location = 'right')
cbar1.formatter = ScalarFormatter(useMathText=True)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
cbar1.locator = MaxNLocator(nbins=4)
cbar1.update_ticks()
axes[0, 0].set_ylabel("Substrate $c_1$")

cbar2 = fig.colorbar(im2, ax=axes[1,:], location = 'right')
cbar3 = fig.colorbar(im3, ax=axes[2,:], location = 'right')
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
axes[1, 0].set_ylabel("Substrate $c_2$")
axes[2, 0].set_ylabel("Substrate $c_3$")

# Tick visibility (same logic, adapted to 3 rows)
for i in (0,1,2):
    axes[i,0].set_xticks([])
    axes[i,1].set_xticks([])
    axes[i,2].set_xticks([])
    axes[i,0].set_yticks([0,0.5,1])

for i in (1,2):
    axes[0,i].set_yticks([])
    axes[1,i].set_yticks([])
    axes[2,i].set_yticks([])
    axes[2,i].set_xticks([0,0.5,1])
axes[2,0].set_xticks([0,0.5,1])

#plt.tight_layout()
plt.show()

## adding individual plots

def format_cbar(cbar):
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

# ---------- Make 2 × len(Tplot) column-wise figures ----------
for it, col in enumerate(Tplot):
    tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"
    
    normP1 = colors.Normalize(vmin=np.min(P1[it]), vmax=np.max(P1[it]))
    normP2 = colors.Normalize(vmin=np.min(P2[it]), vmax=np.max(P2[it]))
    normP3 = colors.Normalize(vmin=np.min(P3[it]), vmax=np.max(P3[it])) 
    # --- Populations figure: 3 rows × 1 col ---
    figP, axesP = plt.subplots(3, 1, figsize=(5, 10), constrained_layout=True, sharex=True)
    imsP = [
        axesP[0].contourf(x, y, P1[it].T, levels=50, cmap='Greens', norm=normP1),
        axesP[1].contourf(x, y, P2[it].T, levels=50, cmap='Oranges', norm=normP2),
        axesP[2].contourf(x, y, P3[it].T, levels=50, cmap='Purples', norm=normP3),
    ]
    axesP[0].set_title(f"{tlabel}")
    axesP[2].set_xlabel("$x$")
    #axesP[:].set_ylabel("$y$")
    #axesP[0].set_ylabel(r"$u_1$")
    #axesP[1].set_ylabel(r"$u_2$")
    #axesP[2].set_ylabel(r"$u_3$")

    # Optional tidy ticks if x,y are in [0,1]
    try:
        axesP[2].set_xticks([0, 0.5, 1])
        for ax in axesP:
            ax.set_yticks([0, 0.5, 1])
    except Exception:
        pass

    # Colourbars (one per subplot, consistent scale across times)
    # Population labels for colourbars
    labels = [r"$u_1$ density", r"$u_2$ density", r"$u_3$ density"]
    
    # Add one colourbar per subplot with appropriate label
    for ax, im, lab in zip(axesP, imsP, labels):
        cb = figP.colorbar(im, ax=ax, location='right')
        format_cbar(cb)
        cb.set_label(lab)

        
    normC1 = colors.Normalize(vmin=np.min(C1[it]), vmax=np.max(C1[it]))
    normC2 = colors.Normalize(vmin=np.min(C2[it]), vmax=np.max(C2[it]))
    normC3 = colors.Normalize(vmin=np.min(C3[it]), vmax=np.max(C3[it]))

    # --- Substrates figure: 3 rows × 1 col ---
    figC, axesC = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=True, sharex=True)
    imsC = [
        axesC[0].contourf(x, y, C1[it].T, levels=50, cmap='Blues', norm=normC1),
        axesC[1].contourf(x, y, C2[it].T, levels=50, cmap='Blues', norm=normC2),
        axesC[2].contourf(x, y, C3[it].T, levels=50, cmap='Blues', norm=normC3),
    ]
    axesC[0].set_title(f"{tlabel}")
    axesC[2].set_xlabel("x")
    axesC[0].set_ylabel(r"$c_1$")
    axesC[1].set_ylabel(r"$c_2$")
    axesC[2].set_ylabel(r"$c_3$")

    try:
        axesC[2].set_xticks([0, 0.5, 1])
        for ax in axesC:
            ax.set_yticks([0, 0.5, 1])
    except Exception:
        pass

    for ax, im in zip(axesC, imsC):
        cb = figC.colorbar(im, ax=ax, location='right')
        format_cbar(cb)

# If running as a script, this will render all 2 × len(Tplot) figures:
plt.show()

# --- Settings you can tweak ---
density_threshold = 0.1   # values <= this are hidden (masked) for all three populations
use_global_scaling = False  # set True to use the same colour scaling across time for each species

# Optional: if you want per-species thresholds, set e.g.
# thr1, thr2, thr3 = 1e-4, 5e-5, 2e-4

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def format_cbar(cbar):
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

# --- (Optional) precompute global colour scaling if desired ---
if use_global_scaling:
    # Flatten across time and space for each species
    P1_all = np.array([np.max(p) for p in P1])
    P2_all = np.array([np.max(p) for p in P2])
    P3_all = np.array([np.max(p) for p in P3])
    # Avoid zero ranges by falling back to 0..1
    normP1_global = colors.Normalize(vmin=0.0, vmax=float(np.max(P1_all)) or 1.0)
    normP2_global = colors.Normalize(vmin=0.0, vmax=float(np.max(P2_all)) or 1.0)
    normP3_global = colors.Normalize(vmin=0.0, vmax=float(np.max(P3_all)) or 1.0)

# Make one figure per time index (column), each with all 3 populations overlaid
for it, col in enumerate(Tplot):
    tlabel = f"t = {Tdata[col]:.2f}" if (0 <= col < len(Tdata)) else f"index {col}"

    # Pick colour scaling
    if use_global_scaling:
        normP1_it, normP2_it, normP3_it = normP1_global, normP2_global, normP3_global
    else:
        # Per-figure (per-time) scaling
        normP1_it = colors.Normalize(vmin=np.min(P1[it]), vmax=np.max(P1[it]) or 1.0)
        normP2_it = colors.Normalize(vmin=np.min(P2[it]), vmax=np.max(P2[it]) or 1.0)
        normP3_it = colors.Normalize(vmin=np.min(P3[it]), vmax=np.max(P3[it]) or 1.0)

    # Mask out low-density values to prevent low-opacity overlap in sparse regions
    # (transpose to match original orientation)
    P1_masked = np.ma.masked_where(P1[it].T <= density_threshold, P1[it].T)
    P2_masked = np.ma.masked_where(P2[it].T <= density_threshold, P2[it].T)
    P3_masked = np.ma.masked_where(P3[it].T <= density_threshold, P3[it].T)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.6), constrained_layout=True)

    # Draw in a stable order; if you want a different visual priority, adjust zorder
    im1 = ax.contourf(x, y, P1_masked, levels=50, cmap='Greens',  norm=normP1_it, zorder=1)
    im2 = ax.contourf(x, y, P2_masked, levels=50, cmap='Oranges', norm=normP2_it, zorder=2)
    im3 = ax.contourf(x, y, P3_masked, levels=50, cmap='Purples', norm=normP3_it, zorder=3)

    ax.set_title(tlabel)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Tidy ticks if domain is [0,1] × [0,1]
    try:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    except Exception:
        pass

    # A simple legend so viewers know which colour is which
    legend_handles = [
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#2ca02c', label=r"$u_1$ (green)"),
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#ff7f0e', label=r"$u_2$ (orange)"),
        Line2D([0], [0], marker='s', linestyle='none', markersize=10, color='#9467bd', label=r"$u_3$ (purple)"),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True)

    # (Optional) If you still want colourbars, uncomment these three lines.
    # They’ll stack colourbars to the right; you can refine placement as needed.
    # cb1 = fig.colorbar(im1, ax=ax, fraction=0.035, pad=0.04); format_cbar(cb1); cb1.set_label(r"$u_1$")
    # cb2 = fig.colorbar(im2, ax=ax, fraction=0.035, pad=0.12); format_cbar(cb2); cb2.set_label(r"$u_2$")
    # cb3 = fig.colorbar(im3, ax=ax, fraction=0.035, pad=0.20); format_cbar(cb3); cb3.set_label(r"$u_3$")

# Render all figures created in the loop
plt.show()


# --- Adjustable settings ---
density_threshold = 0.07   # Mask regions where densities are below this value
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
    cb = fig.colorbar(im3, ax=ax, fraction=0.04, pad=0.04)
    #cb.set_label(r"$u_1$ density")
    format_cbar(cb)


# Show all figures
plt.show()
