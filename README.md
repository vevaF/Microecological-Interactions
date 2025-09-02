# Multi-species Bacterial Dynamics — Competition & Commensalism

This repository contains Python implementations of **2D diffusion–advection–reaction** models for multi‑species bacterial systems with **pressure‑driven movement** and **nutrient dynamics**, under two ecological scenarios:

1. **Competition** — three populations compete for *space* (via pressure) and a *shared nutrient*.
2. **Commensalism** — a chain where species transform substrates used by the next species.

Both solvers use a method‑of‑lines discretisation and `scipy.integrate.solve_ivp` for time integration.

---

## Repository structure

```
.
├── competition.py      # Competition: 3 species, 1 shared substrate
├── commensalism.py     # Commensalism: 3 species, 3 linked substrates (C1→C2→C3)
└── README.md
```

---

## 1) Competition model

### State variables
- $P_i(x,t)$ for $i=1,2,3$: bacterial densities (one per species).  
- $C(x,t)$: shared substrate (nutrient) density.  
- **Domain:** $x=(x_1,x_2)\in[0,L_x]\times[0,L_y]$ with **no‑flux** boundaries.

### Population pressure and movement
- **Pressure:** $P_r=P_1+P_2+P_3$.  
- **Negative chemotaxis:** drift **down** the pressure gradient (repulsion). In flux form this is represented by
  $$
  \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big),
  $$
  where $D_i$ is the diffusion coefficient and $A_i$ the pressure sensitivity.

### Governing equations
**Bacteria ($i=1,2,3$):**
$$
\frac{\partial P_i}{\partial t}
= \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big) + F_i(P_1,P_2,P_3,C).
$$

**Substrate:**
$$
\frac{\partial C}{\partial t}
= d\,\Delta C + G(P_1,P_2,P_3,C).
$$

- $F_i$ encodes **birth** and **death** terms (competition for substrate is included here).  
- $G$ encodes **consumption** of the substrate by the populations.  
- Boundary conditions are **no‑flux** on the rectangular domain.

> **Note:** The README intentionally keeps $F_i$ and $G$ generic to reflect your corrected specification. Choose the exact kinetics (e.g. logistic crowding, Monod uptake) in the code as needed.

---

## 2) Commensalism model

### State variables
- $P_1,P_2,P_3$: bacterial densities.  
- $C_1,C_2,C_3$: substrates in a **commensal chain**.

### Ecological chain
- $P_1$ **consumes** $C_1$ and **produces** $C_2$.  
- $P_2$ **consumes** $C_2$ and **produces** $C_3$.  
- $P_3$ **consumes** $C_3$.

### Movement and pressure
- Same movement law and pressure as in Competition: $P_r=P_1+P_2+P_3$ with drift down $\nabla P_r$.

### Governing equations
**Bacteria ($i=1,2,3$):**
$$
\frac{\partial P_i}{\partial t}
= \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big) + F_i \big(P_i,C_i\big).
$$

**Substrates:**
$$
\begin{aligned}
\frac{\partial C_1}{\partial t} &= d\,\Delta C_1 + G_1(P_1,C_1)\quad\text{(consumption by $P_1$)},\\[2pt]
\frac{\partial C_2}{\partial t} &= d\,\Delta C_2 + G_2(P_1,P_2,C_1,C_2)\quad\text{(produced by $P_1$, consumed by $P_2$)},\\[2pt]
\frac{\partial C_3}{\partial t} &= d\,\Delta C_3 + G_3(P_2,P_3,C_2,C_3)\quad\text{(produced by $P_2$, consumed by $P_3$)}.
\end{aligned}
$$

- $F_i$ handle species‑specific birth/death.  
- $G_1,G_2,G_3$ encode the **production/consumption** pattern described above.  
- Domain and boundary conditions: **no‑flux**.

> As above, leave $F_i$ and $G_j$ in the scripts as explicit functions you can tailor (e.g. Monod uptake with yields, linear production, etc.).

---

## Running the code

### Requirements
- Python ≥ 3.9  
- NumPy, SciPy, Matplotlib

Install:

```bash
pip install numpy scipy matplotlib
```

### Execute
```bash
python competition.py     # Competition scenario
python commensalism.py    # Commensalism scenario
```

Each script:
- Defines grid, parameters, and initial conditions.  
- Integrates with `solve_ivp`.  
- Produces contour plots of $P_i$ and $C$ (competition) or $C_j$ (commensalism).

---

## Tips and notes

- **Stability:** start with moderate time horizons and grid sizes; refine as needed.  
- **Parameters:** tune $D_i$, $A_i$, and the kinetics in $F_i, G$/$G_j$ to explore regimes.  
- **Boundaries:** defaults are no‑flux; adapt to Dirichlet/periodic if desired.  
- **Performance:** vectorise fluxes and consider stiff integrators (`LSODA`, `BDF`) for demanding settings.


## Contact

V. Freingruber: v.e.freingruber@tudelft.nl

