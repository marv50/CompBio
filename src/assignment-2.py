import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ode_sys(t,x):
    rna_a, rna_b, p_a, p_b = x

    theta_a = 0.21
    theta_b = 0.21
    n_a = 3
    n_b = 3
    max_a = 2.35
    max_b = 2.35
    gamma_a = 1
    gamma_b = 1
    k_a = 1.0
    k_b = 1.0
    delta_a = 1.0
    delta_b = 1.0

    hact = p_b**n_b/(p_b**n_b+theta_b**n_b) 
    hin = theta_a**n_a/(p_a**n_a+theta_a**n_a)

    rna_a = max_a * hact - gamma_a*rna_a
    rna_b  = max_b * hin - gamma_b*rna_b
    p_a = k_a*rna_a - delta_a*p_a
    p_b = k_b*rna_b - delta_b*p_b

    return [rna_a, rna_b, p_a, p_b]

t_span = [0,1000]
y0 = [0.8, 0.8, 0.8, 0.8]

sol = solve_ivp(ode_sys, t_span, y0)
y = sol.y
t = sol.t

plt.figure()
plt.plot(t, y[0,:], label = "rna A")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.show()