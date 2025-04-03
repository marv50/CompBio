import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import scipy.optimize as sp

import numpy as np
from matplotlib.pyplot import *
import scipy.optimize as sp

#%% Generate Synthetic Data
np.random.seed(42)  # For reproducibility
x = np.sort(np.random.uniform(0, 10, 20))  # Random substrate concentrations between 0 and 10
y = 2 * x + np.sin(2 * np.pi * x / 10) + np.random.normal(0, 0.2, x.shape)  # Linear + sine + noise

figure()
scatter(x, y, label='Data points')  # Use scatter for points
legend()
xlabel('y [units]')
ylabel('x [units]')
title('Data Points')
show()






#%% Define Linear Function
def linear(x, a, b):
    return a * x + b

# Fit Linear Function
params_linear, _ = sp.curve_fit(linear, x, y)
a_fit, b_fit = params_linear

x_fit = np.linspace(0, 10, 100)

figure()
scatter(x, y, label='Data points')  # Use scatter for points
plot(x_fit, linear(x_fit, *params_linear), color='red', label='Linear Fit')
legend()
xlabel('x [units]')
ylabel('y [units]')
title('Linear Fit')
show()




#%% Define Linear + Sine Function
def linear_sine(x, a, b, A, freq):
    return a * x + b + A * np.sin(freq * x)

# Fit Linear + Sine Function
params_lin_sine, _ = sp.curve_fit(linear_sine, x, y, p0=[a_fit, b_fit, 0.1, 2 * np.pi / 10])
a_fit_ls, b_fit_ls, A_fit, freq_fit = params_lin_sine

figure()
scatter(x, y, label='Data points')  # Use scatter for points
plot(x_fit, linear(x_fit, *params_linear), color='red', label='Linear Fit')
plot(x_fit, linear_sine(x_fit, *params_lin_sine), color='blue', label='Linear + Sine Fit')
legend()
xlabel('x [units]')
ylabel('y [units]')
title('Linear vs. Linear + Sine Fit')
show()





#%% Goodness of Fit Metrics
def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def chi_squared(y, y_fit):
    return np.sum(((y - y_fit) ** 2) / y_fit)

r2_linear = r_squared(y, linear(x, *params_linear))
chi_linear = chi_squared(y, linear(x, *params_linear))

r2_lin_sine = r_squared(y, linear_sine(x, *params_lin_sine))
chi_lin_sine = chi_squared(y, linear_sine(x, *params_lin_sine))

fit_results = {
    "Linear": {"R2": r2_linear, "Chi2": chi_linear},
    "Linear + Sine": {"R2": r2_lin_sine, "Chi2": chi_lin_sine},
}

# Print results
print("Linear Fit:")
print(f"  R-squared: {r2_linear:.4f}")
print(f"  Chi-squared: {chi_linear:.4f}")

print("\nLinear + Sine Fit:")
print(f"  R-squared: {r2_lin_sine:.4f}")
print(f"  Chi-squared: {chi_lin_sine:.4f}")





#%% Load Data from CSV
data = pd.read_csv("fixed_data.csv")
S = data["S"].values
v = data["v"].values


figure()
scatter(S, v, label='Data points')# Use scatter for points
legend()
xlabel('S [units]')
ylabel('v [units]')
title('Data Points')
show()



#%%
# Define the Michaelis-Menten function
def michaelis_menten(S, V_max, K_m):
    return (V_max * S) / (K_m + S)

# Perform curve fitting
params, covariance = sp.curve_fit(michaelis_menten, S, v, bounds=([0.1, 0.1], [10.1, 10.1]))

# Extract estimated V_max and K_m
V_max_fit, K_m_fit = params

print(f"Estimated V_max: {V_max_fit}")
print(f"Estimated K_m: {K_m_fit}")

S_fit = np.linspace(0,10,100)

figure()
scatter(S, v, label='Data points')# Use scatter for points
plot(S_fit, michaelis_menten(S_fit, V_max_fit, K_m_fit), color='red', label='Fitted curve')
legend()
xlabel('S [units]')
ylabel('v [units]')
title('Michaelis-Menten Fit')
show()



#%%
# Compute R-squared
def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Compute Chi-squared
def chi_squared(y, y_fit):
    return np.sum(((y - y_fit) ** 2) / y_fit)

r2_mm = r_squared(v, michaelis_menten(S, V_max_fit, K_m_fit))
chi_mm = chi_squared(v, michaelis_menten(S, V_max_fit, K_m_fit))

fit_results = {}
fit_results = {
    "Michaelis-Menten": {"R2": r2_mm, "Chi2": chi_mm},
}

# Print results
print(f"R-squared: {r2_mm}")
print(f"Chi-squared: {chi_mm}")






#%%
# Define the Hill equation
def hill_equation(S, V_max, K_m, n):
    return (V_max * S**n) / (K_m**n + S**n)

# Perform curve fitting
params_hill, covariance_hill = sp.curve_fit(hill_equation, S, v, bounds=([0.1, 0.1, 0.1], [10.1, 10.1, 10.1]))

# Extract estimated parameters
V_max_fit_hill, K_m_fit_hill, n_fit_hill = params_hill

print(f"Estimated V_max: {V_max_fit_hill}")
print(f"Estimated K_m: {K_m_fit_hill}")
print(f"Estimated Hill coefficient (n): {n_fit_hill}")

figure()
scatter(S, v, label='Data points')  # Data as points
plot(S_fit, hill_equation(S_fit, *params_hill), label='Fitted curve', color='orange')  # Fitted curve
xlabel('S')
ylabel('v')
title('Hill Equation Fit')
legend()
show()
