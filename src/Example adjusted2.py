import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import scipy.optimize as sp

# Load Data from CSV
data = pd.read_csv("Compbio/data/Kinetics.csv")
S1 = data["S1"].values
S2 = data["S2"].values
v = data["Rate"].values

# Define Goodness-of-Fit Metrics
def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def chi_squared(y, y_fit):
    return np.sum(((y - y_fit) ** 2) / y_fit)

# Define Enzyme Kinetics Models
# Type 1a: Ordered Bi-Bi
def rate_type1a(S, Vmax, Kis1, Km1):
    S1, S2 = S
    return (Vmax * S1 * S2) / (Kis1 * Km1 + Km1 * S1 + S1 * S2)

# Type 1b: Random Bi-Bi
def rate_type1b(S, Vmax, Km1, Km2):
    S1, S2 = S
    return (Vmax * S1 * S2) / (Km1 * Km2 + Km2 * S1 + Km1 * S2 + S1 * S2)

# Type 2: Ping Pong
def rate_type2(S, Vmax, Km1, Km2):
    S1, S2 = S
    return (Vmax * S1 * S2) / (Km1 * S2 + Km2 * S1 + S1 * S2)

# Define the Michaelis-Menten function
def michaelis_menten(S, V_max, K_m):
    return (V_max * S) / (K_m + S)

# Define the Hill equation
def hill_equation(S, V_max, K_m, n):
    return (V_max * S**n) / (K_m**n + S**n)


# Fit Models and Compare
models = {
    "Type 1a (Ordered)": (rate_type1a, [1.0, 1.0, 1.0]),
    "Type 1b (Random)": (rate_type1b, [1.0, 1.0, 1.0]),
    "Type 2 (Ping-Pong)": (rate_type2, [1.0, 1.0, 1.0]),
    
}

fit_results = {}

for name, (model_func, p0) in models.items():
    try:
        # Fit the model
        params, _ = sp.curve_fit(model_func, (S1, S2), v, p0=p0, bounds=(0, np.inf))
        v_fit = model_func((S1, S2), *params)
        
        # Compute goodness-of-fit metrics
        r2 = r_squared(v, v_fit)
        chi2 = chi_squared(v, v_fit)
        
        # Store results
        fit_results[name] = {"params": params, "R2": r2, "Chi2": chi2}
        
        # Plot observed vs predicted rates
        figure()
        scatter(v, v_fit, label="Predicted vs Observed")
        plot([v.min(), v.max()], [v.min(), v.max()], 'k--', label="Perfect Fit")
        xlabel("Observed Rate")
        ylabel("Predicted Rate")
        title(f"Model Fit: {name}")
        legend()
        show()
    except Exception as e:
        fit_results[name] = {"params": None, "R2": None, "Chi2": None, "Error": str(e)}

# Print the Summary of Results
for name, result in fit_results.items():
    print(f"\n{name}")
    if result["params"] is not None:
        print(f"  Parameters: {result['params']}")
        print(f"  R²: {result['R2']:.4f}")
        print(f"  χ²: {result['Chi2']:.4e}")
    else:
        print(f"  Fit failed: {result.get('Error', 'Unknown error')}")