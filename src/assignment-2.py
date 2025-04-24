import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE system with external parameters
def ode_sys(t, x, params):
    rna_a, rna_b, p_a, p_b = x
    
    # Extract parameters
    theta_a = params["theta_a"]
    theta_b = params["theta_b"]
    n_a = params["n_a"]
    n_b = params["n_b"]
    max_a = params["max_a"]
    max_b = params["max_b"]
    gamma_a = params["gamma_a"]
    gamma_b = params["gamma_b"]
    k_a = params["k_a"]
    k_b = params["k_b"]
    delta_a = params["delta_a"]
    delta_b = params["delta_b"]
    
    # Regulatory functions
    hact = p_b**n_b / (p_b**n_b + theta_b**n_b)
    hin = theta_a**n_a / (p_a**n_a + theta_a**n_a)
    
    # Derivatives
    drna_a = max_a * hact - gamma_a * rna_a
    drna_b = max_b * hin - gamma_b * rna_b
    dp_a = k_a * rna_a - delta_a * p_a
    dp_b = k_b * rna_b - delta_b * p_b

    return [drna_a, drna_b, dp_a, dp_b]

# Define parameters
params = {
    "theta_a": 0.21, "theta_b": 0.21,
    "n_a": 3, "n_b": 3,
    "max_a": 2.35, "max_b": 2.35,
    "gamma_a": 1.0, "gamma_b": 1.0,
    "k_a": 1.0, "k_b": 1.0,
    "delta_a": 1.0, "delta_b": 1.0
}

# Initial conditions and time span
y0 = [0.8, 0.8, 0.8, 0.8]
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 50)

# Solve the ODE system
sol = solve_ivp(lambda t, x: ode_sys(t, x, params), t_span, y0, t_eval=t_eval)

# Extract solution
t = sol.t
rna_a, rna_b, p_a, p_b = sol.y

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(t, rna_a, label="RNA A", linewidth=2)
plt.plot(t, rna_b, label="RNA B", linewidth=2)
plt.plot(t, p_a, label="Protein A", linestyle='--', linewidth=2)
plt.plot(t, p_b, label="Protein B", linestyle='--', linewidth=2)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Concentration", fontsize=12)
plt.title("Gene Regulation Dynamics", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
