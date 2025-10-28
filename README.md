# Multi-species Bacterial Dynamics — Competition & Commensalism

This repository contains Python implementations of **2D cross-diffusion–reaction** models for multi‑species bacterial systems with **pressure‑driven movement** and **nutrient dynamics**, under three ecological scenarios:

1. **Competition** — three populations compete for a *shared nutrient* and for *space* (via pressure).
2. **Competition** — two populations compete for a *shared nutrient* and for *space* (via pressure), and they work with different growth strategies (yield vs. growth strategist).
3. **Commensalism** — a food chain where species transform substrates used by the next species.

Both solvers use a method‑of‑lines discretisation and `scipy.integrate.solve_ivp` for time integration.
More information on the numerical implementation can be found in the appendix of the paper [link].

---

## Content of repository

```
.
├── competition3.py      # Competition: 3 species, 1 shared substrate
├── competition2.py      # Competition: 2 species, 1 shared substrate, different growth strategies
├── commensalism.py     # Commensalism: 3 species, 3 linked substrates (C1→C2→C3)
└── README.md
```

---
## 1) Competition model for three populations

### State variables
- $P_i(x,t)$ for $i=1,2,3$: bacterial densities (one per species).  
- $C(x,t)$: shared substrate (nutrient) density.  
- **Domain:** $x=(x_1,x_2)\in[0,L_x]\times[0,L_y]$ with **no‑flux** boundaries for the bacterial population and Neumann boundary conditions for the substrate.

### Population movement
- **Diffusion**
- **Pressure-driven directed movement:** drift **down** the gradient of the total population density, where the pressure is defined as $P_r:=\alpha (P_1+P_2+P_3)$.
- In flux form this is represented by
$$
  \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big),
$$
  for the $i$-th population, where $D_i$ is the diffusion coefficient and $A_i$ the sensitivity to the pressure.

### Substrate movement
- **Diffusion**

### Population growth and decay:
- **Growth** by substrate consumption (e.g. Monod kinetics)
- **Decay**: Density-dependent decay term
- Here: For $i$-th population the net growth rate is $F_i = r_i \frac{C}{K_i + C} - b_i Pr$, where $r_i$ is the max. growth rate for the $i$-th population, $K_i$ is the half saturation concentration and $b_i$ is the decay rate.

### Substrate uptake:
- **Uptake** by bacterial populations (e.g. Monod kinetics)
- Here: $G(P1,P2,P3,C):= \sum_{i=1}^3 \frac{r_i}{Y_i} \frac{C}{K_i + C}$, where $Y_i$ is the yield coefficient.

### Governing equations
**Bacteria ($i=1,2,3$):**

$$
\frac{\partial P_i}{\partial t}
= \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big) + F_i(P_1,P_2,P_3,C).
$$

**Substrate:**

$$
\frac{\partial C}{\partial t}
= d\,\Delta C - G(P_1,P_2,P_3,C).
$$

---

## 1) Competition model for two populations

### Similar to 1) with i=1,2
> **Note:** Different initial setup than in the previous competition case.

---

## 3) Commensalism model

### State variables
- $P_i(x,t)$ for $i=1,2,3$: bacterial densities.  
- $C_j(x,t)$ for $j=1,2,3$: substrate densities.  
- **Domain:** $x=(x_1,x_2)\in[0,L_x]\times[0,L_y]$ with **no‑flux** boundaries for the bacterial population and Neumann boundary conditions for the substrate $C_1$.
- 
### Ecological chain
- $P_1$ **consumes** $C_1$ and **produces** $C_2$.  
- $P_2$ **consumes** $C_2$ and **produces** $C_3$.  
- $P_3$ **consumes** $C_3$.

### Population + Substrate movement
- see 1)

### Population growth and decay:
- **Growth** by substrate consumption (e.g. Monod kinetics)
- **Decay**: Density-dependent decay term
- Here: For $i$-th population the net growth rate is $F_i = r_i \frac{C}{K_i + C_i} - b_i Pr$, where $r_i$ is the max. growth rate for the $i$-th population, $K_i$ is the half saturation concentration and $b_i$ is the decay rate.
- Difference to competition models: Every population has their own substrate to grow on.

### Substrate uptake and production:
- **Uptake** by a bacterial population (e.g. Monod kinetics)
- Here: For the $j$-th substrate, $j=1,2,3$, the uptake term is $G_j(P_j,C_j):=  \frac{r_j}{Y_j} \frac{C_j}{K_j + C_j}$, where $Y_j$ is the yield coefficient.
- **Production** as a metabolic byproduct
- Here: The substrate $C_2$ is a byproduct of the metabolic conversion of $C_1$ by $P_1$, i.e. $H_2(P_1, C_1) = \sigma \frac{r_1}{Y_1} \frac{C_1}{K_1 + C_1}$ with conversion factor $\sigma$; similarily the substrate is a byproduct byproduct of the metabolic conversion of $C_2$ by $P_2$, i.e. $H_3(P_2, C_2) = \sigma \frac{r_2}{Y_2} \frac{C_2}{K_2 + C_2}$.

### Governing equations
**Bacteria ($i=1,2,3$):**

$$
\frac{\partial P_i}{\partial t}
= \nabla \cdot \big(D_i \nabla P_i - A_i P_i \nabla P_r \big) + F_i \big(P_i,C_i\big).
$$

**Substrates:**

$$
\begin{aligned}
\frac{\partial C_1}{\partial t} &= d\,\Delta C_1 - G_1(P_1,C_1),\\[2pt]
\frac{\partial C_2}{\partial t} &= d\,\Delta C_2 - G_2(P_2,C_2) + H_2(P_1, C_1),\\[2pt]
\frac{\partial C_3}{\partial t} &= d\,\Delta C_3 - G_3(P_3,C_3) + H_3(P_2, C_2) .
\end{aligned}
$$

## Contact

V. Freingruber: v.e.freingruber@tudelft.nl

