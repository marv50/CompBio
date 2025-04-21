"""
Plots_S2.py

This script is designed to analyze enzyme kinetics data and generate visualizations for specific S2 values.
It performs the following tasks:
1. Loads enzyme kinetics data from a CSV file.
2. Fits the Type 2 (Ping-Pong) enzyme kinetics model to the data.
3. Interpolates S1 and rate values for specific S2 values.
4. Generates Eadie-Hofstee and Lineweaver-Burk plots for the interpolated data.
5. Extracts Michaelis constants (Km1, Km2) and Vmax from the Eadie-Hofstee plot.
6. Saves the interpolated results to a CSV file.

Functions:
- `rate_type2`: Defines the Type 2 (Ping-Pong) enzyme kinetics model.
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import scipy.optimize as sp
from scipy.optimize import fsolve
from scipy.stats import linregress

# Load Data from CSV
data = pd.read_csv("Compbio/data/Kinetics.csv")
S1 = data["S1"].values
S2 = data["S2"].values
v = data["Rate"].values

# Define Type 2 rate equation
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

# Fit the Type 2 model to get parameters
params_type2, _ = sp.curve_fit(rate_type2, (S1, S2), v, p0=[1.0, 1.0, 1.0], bounds=(0, np.inf))
Vmax_fit, Km1_fit, Km2_fit = params_type2

# Define specific S2 values for which to interpolate S1 and rate
specific_S2_values = [1.5, 2.5, 5]

# Interpolate S1 and rate values for the specific S2 values
interpolated_results = {S2_val: {"S1": [], "Rate": []} for S2_val in specific_S2_values}

for S2_val in specific_S2_values:
    for rate_val in v:
        # Solve for S1 using the fitted model
        def solve_for_S1(S1):
            """
            Solves for S1 given S2, rate, and the fitted Type 2 model parameters.

            Parameters:
            - S1: Substrate concentration S1 (float).

            Returns:
            - Difference between the calculated and observed rate (float).
            """
            return rate_type2((S1, S2_val), Vmax_fit, Km1_fit, Km2_fit) - rate_val
        
        S1_initial_guess = 0.1  # Initial guess for S1
        S1_solution = fsolve(solve_for_S1, S1_initial_guess)[0]
        interpolated_results[S2_val]["S1"].append(S1_solution)
        interpolated_results[S2_val]["Rate"].append(rate_val)

# Generate a single Eadie-Hofstee plot for all S2 values
# Combine data for all S2 values
all_v = []
all_v_over_S1 = []
all_labels = []

for S2_val in specific_S2_values:
    S1_vals = interpolated_results[S2_val]["S1"]
    rate_vals = interpolated_results[S2_val]["Rate"]
    v_over_S1 = [rate / S1 for rate, S1 in zip(rate_vals, S1_vals)]
    
    all_v.extend(rate_vals)
    all_v_over_S1.extend(v_over_S1)
    all_labels.extend([S2_val] * len(rate_vals))  # Label each point with its S2 value

# Create a single Eadie-Hofstee plot
figure()
for S2_val in specific_S2_values:
    # Filter points for the current S2 value
    v_vals = [v for v, label in zip(all_v, all_labels) if label == S2_val]
    v_over_S1_vals = [v_over for v_over, label in zip(all_v_over_S1, all_labels) if label == S2_val]
    
    # Plot points for the current S2 value
    scatter(v_over_S1_vals, v_vals, label=f"S2 = {S2_val}", marker ='.', s=50, alpha=0.7)

# Add labels, title, and legend
xlabel("v / [S1]")
ylabel("v")
title("Eadie-Hofstee Plot for S2 Values")
legend()
grid()
show()

# Extract Michaelis constants and Vmax from the Eadie-Hofstee plot
results = []

for S2_val in specific_S2_values:
    # Get interpolated S1 and rate values for the current S2 value
    S1_vals = interpolated_results[S2_val]["S1"]
    rate_vals = interpolated_results[S2_val]["Rate"]

    # Set fixed S1 value for Km2 calculation
    S1_fixed = 1.0  
    
    # Calculate v / [S1] for the Eadie-Hofstee plot
    v_over_S1 = [rate / S1 for rate, S1 in zip(rate_vals, S1_vals)]
    
    # Perform linear regression (v vs. v / [S1])
    slope, intercept, r_value, p_value, std_err = linregress(v_over_S1, rate_vals)
    
    # Extract Vmax and Km1
    Vmax = intercept  # y-intercept
    Km1 = -slope      # Negative slope

    # Estimate v using the Type 2 model equation and fitted value for Km2
    v_fixed = (Vmax * S1_fixed * S2_val) / (Km1 * S2_val + Km2_fit * S1_fixed + S1_fixed * S2_val)
    print(Km1)

    # Calculate Km2 using the Type 2 model equation
    Km2 = (Vmax * S1_fixed * S2_val - v_fixed * Km1 * S2_val - v_fixed * S1_fixed * S2_val) / (v_fixed * S1_fixed) # Rewritten v equation for Km2
    print(Km2)

    # Store results
    results.append({"S2": S2_val, "Vmax": Vmax, "Km1": Km1, "Km2": Km2})

# Print the extracted parameters
print("\nExtracted Michaelis Constants and Vmax:")
for result in results:
    print(f"S2 = {result['S2']:.2f} nM: Vmax = {result['Vmax']:.4f}, Km1 = {result['Km1']:.4f}, Km2 = {result['Km2']:.4f}")

# Generate a single Lineweaver-Burk plot for all S2 values
figure()

for S2_val in specific_S2_values:
    # Get interpolated S1 and rate values for the current S2 value
    S1_vals = interpolated_results[S2_val]["S1"]
    rate_vals = interpolated_results[S2_val]["Rate"]
    
    # Calculate 1 / [S1] and 1 / v
    reciprocal_S1 = [1 / S1 for S1 in S1_vals]
    reciprocal_v = [1 / v for v in rate_vals]
    
    # Plot points for the current S2 value
    scatter(reciprocal_S1, reciprocal_v, label=f"S2 = {S2_val}", marker='.', s=50, alpha=0.7)

# Add labels, title, and legend
xlabel("1 / [S1]")
ylabel("1 / v")
title("Lineweaver-Burk Plot for S2 Values")
legend()
grid()
show()

# Save interpolated results to a new CSV file
output_data = []

for S2_val in specific_S2_values:
    S1_vals = interpolated_results[S2_val]["S1"]
    rate_vals = interpolated_results[S2_val]["Rate"]
    for S1, rate in zip(S1_vals, rate_vals):
        output_data.append({"S1": S1, "Rate": rate, "S2": S2_val})


# Convert to DataFrame and save as CSV
output_df = pd.DataFrame(output_data)
output_df.to_csv("Compbio/data/Interpolated_Kinetics.csv", index=False)
print("Interpolated data saved to 'Compbio/data/Interpolated_Kinetics.csv'")

