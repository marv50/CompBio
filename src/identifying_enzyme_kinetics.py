"""
Identifying_enzyme_kinetics.py

This script is designed to analyze enzyme kinetics data using various models, including:
- Type 1a (Ordered Bi-Bi)
- Type 1b (Random Bi-Bi)
- Type 2 (Ping-Pong)

The script performs the following tasks:
1. Loads enzyme kinetics data from a CSV file.
2. Defines enzyme kinetics models and goodness-of-fit metrics.
3. Fits the models to the data using non-linear regression.
4. Evaluates the models using R² and χ² metrics.
5. Visualizes the observed vs. predicted rates for each model.
6. Saves the best-fitting model's plot and outputs the parameters.

Functions:
- `r_squared`: Computes the coefficient of determination (R²) for model evaluation.
- `chi_squared`: Computes the chi-squared statistic for model evaluation.
- `rate_type1a`: Defines the Type 1a (Ordered Bi-Bi) enzyme kinetics model.
- `rate_type1b`: Defines the Type 1b (Random Bi-Bi) enzyme kinetics model.
- `rate_type2`: Defines the Type 2 (Ping-Pong) enzyme kinetics model.
- `michaelis_menten`: Defines the Michaelis-Menten equation.
- `hill_equation`: Defines the Hill equation for cooperative enzyme kinetics.
"""

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
    """
    Computes the coefficient of determination (R²) to evaluate model fit.

    Parameters:
    - y: Observed data (array-like).
    - y_fit: Predicted data from the model (array-like).

    Returns:
    - R² value (float).
    """
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def chi_squared(y, y_fit):
    """
    Computes the chi-squared statistic to evaluate model fit.

    Parameters:
    - y: Observed data (array-like).
    - y_fit: Predicted data from the model (array-like).

    Returns:
    - Chi-squared value (float).
    """
    return np.sum(((y - y_fit) ** 2) / y_fit)

# Define Enzyme Kinetics Models
# Type 1a: Ordered Bi-Bi
def rate_type1a(S, Vmax, Kis1, Km1):
    """
    Defines the Type 1a (Ordered Bi-Bi) enzyme kinetics model.

    Parameters:
    - S: Tuple of substrate concentrations (S1, S2).
    - Vmax: Maximum reaction rate.
    - Kis1: Inhibition constant for S1.
    - Km1: Michaelis constant for S1.

    Returns:
    - Reaction rate (float).
    """
    S1, S2 = S
    return (Vmax * S1 * S2) / (Kis1 * Km1 + Km1 * S1 + S1 * S2)

# Type 1b: Random Bi-Bi
def rate_type1b(S, Vmax, Km1, Km2):
    """
    Defines the Type 1b (Random Bi-Bi) enzyme kinetics model.

    Parameters:
    - S: Tuple of substrate concentrations (S1, S2).
    - Vmax: Maximum reaction rate.
    - Km1: Michaelis constant for S1.
    - Km2: Michaelis constant for S2.

    Returns:
    - Reaction rate (float).
    """
    S1, S2 = S
    return (Vmax * S1 * S2) / (Km1 * Km2 + Km2 * S1 + Km1 * S2 + S1 * S2)

# Type 2: Ping Pong
def rate_type2(S, Vmax, Km1, Km2):
    """
    Defines the Type 2 (Ping-Pong) enzyme kinetics model.

    Parameters:
    - S: Tuple of substrate concentrations (S1, S2).
    - Vmax: Maximum reaction rate.
    - Km1: Michaelis constant for S1.
    - Km2: Michaelis constant for S2.

    Returns:
    - Reaction rate (float).
    """
    S1, S2 = S
    return (Vmax * S1 * S2) / (Km1 * S2 + Km2 * S1 + S1 * S2)

# Define the Michaelis-Menten function
def michaelis_menten(S, V_max, K_m):
    """
    Defines the Michaelis-Menten equation for enzyme kinetics.

    Parameters:
    - S: Substrate concentration.
    - V_max: Maximum reaction rate.
    - K_m: Michaelis constant.

    Returns:
    - Reaction rate (float).
    """
    return (V_max * S) / (K_m + S)

# Define the Hill equation
def hill_equation(S, V_max, K_m, n):
    """
    Defines the Hill equation for cooperative enzyme kinetics.

    Parameters:
    - S: Substrate concentration.
    - V_max: Maximum reaction rate.
    - K_m: Michaelis constant.
    - n: Hill coefficient (degree of cooperativity).

    Returns:
    - Reaction rate (float).
    """
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

        # Save the plot for the "Type 2 (Ping-Pong)" model
        if name == "Type 2 (Ping-Pong)":
            savefig("Type2_PingPong_Model_Fit.pdf")

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
        print(f" Fit failed: {result.get('Error', 'Unknown error')}")