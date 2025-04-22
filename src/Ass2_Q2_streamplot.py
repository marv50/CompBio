'''
# Gene regulation dynamics Streamplot'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 2
beta = 1.1
gamma = 1
delta = 0.9

# Vector field functions
def f(x, y):
    return alpha * x - beta * x * y

def g(x, y):
    return -gamma * y + delta * x * y

# Meshgrid
x = np.linspace(0.01, 2.5, 100)
y = np.linspace(0.01, 2.5, 100)
X, Y = np.meshgrid(x, y)

# Vector field
Fx = f(X, Y)
Fy = g(X, Y)

# Angle and magnitude
theta = np.arctan2(Fy, Fx)
magnitude = np.sqrt(Fx**2 + Fy**2)

# Plotting
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, Fx, Fy, color='gray', density=1.2)

# Nullclines
y_null = alpha / beta
x_null = gamma / delta
plt.axhline(y=y_null, color='red', linestyle='--', label='dx/dt = 0')
plt.axvline(x=x_null, color='blue', linestyle='--', label='dy/dt = 0')
plt.axhline(y=0, color='red', linestyle=':', linewidth=1)
plt.axvline(x=0, color='blue', linestyle=':', linewidth=1)


# Equilibria
plt.plot(0, 0, 'ko', label='Equilibrium at (0,0)')
plt.plot(x_null, y_null, 'mo', label='Equilibrium at (γ/δ, α/β)')

# Labels
plt.xlabel("x (Metabolite)", fontsize=12)
plt.ylabel("y (Enzyme)", fontsize=12)
plt.title("Stability Streamplot of Lotka-Volterra Dynamics", fontsize=14)
plt.xlim(0, 2.5)
plt.ylim(0, 2.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.grid(True, which='both', linestyle=':', linewidth=0.7)
plt.savefig('streamplot.pdf', dpi=300, bbox_inches='tight')
plt.show()
