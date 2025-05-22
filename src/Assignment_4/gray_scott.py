import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_grid(N):
    u = np.ones((N, N), dtype=np.float64) * 0.5
    v = np.zeros((N, N), dtype=np.float64)

    center = N // 2
    size = N // 10
    v[center - size:center + size, center - size:center + size] = 0.25

    noise_u = np.random.uniform(-0.01, 0.01, (N, N))
    noise_v = np.random.uniform(-0.01, 0.01, (N, N))
    u += noise_u
    v += noise_v

    return u, v

def apply_boundary_conditions(grid, bc_type, alpha=0.5, beta=1.0, gamma=0.0):
    if bc_type == "periodic":
        grid[0, :] = grid[-2, :]
        grid[-1, :] = grid[1, :]
        grid[:, 0] = grid[:, -2]
        grid[:, -1] = grid[:, 1]
    elif bc_type == "dirichlet":
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
    elif bc_type == "neumann":
        grid[0, :] = grid[1, :]
        grid[-1, :] = grid[-2, :]
        grid[:, 0] = grid[:, 1]
        grid[:, -1] = grid[:, -2]
    elif bc_type == "robin":
        grid[0, :] = (gamma - beta * grid[1, :]) / alpha
        grid[-1, :] = (gamma - beta * grid[-2, :]) / alpha
        grid[:, 0] = (gamma - beta * grid[:, 1]) / alpha
        grid[:, -1] = (gamma - beta * grid[:, -2]) / alpha

def compute_laplacian(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid
    )

# Parameters
N = 200
Du, Dv = 0.16, 0.08
f, k = 0.035, 0.060
dt = 1.0
steps = 10000
bc_type = "dirichlet"

# Initialize grids
u, v = initialize_grid(N)

# Plot initial condition before animation
plt.figure(figsize=(8, 8))
plt.imshow(v, cmap='inferno', interpolation='bilinear')
plt.title(f'Initial Condition ({bc_type} BC)')
plt.colorbar()
plt.show()


# Set up figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
im = ax.imshow(v, cmap='inferno', interpolation='bilinear')
ax.set_title(f'Gray-Scott Model ({bc_type} BC)')

def update(frame):
    global u, v
    for _ in range(10):  # Advance 10 simulation steps per frame to speed things up
        lap_u = compute_laplacian(u)
        lap_v = compute_laplacian(v)

        uvv = u * v * v
        u += (Du * lap_u - uvv + f * (1 - u)) * dt
        v += (Dv * lap_v + uvv - (f + k) * v) * dt

        apply_boundary_conditions(u, bc_type)
        apply_boundary_conditions(v, bc_type)

    im.set_array(v)
    ax.set_xlabel(f'Time step: {frame * 10}')
    return [im]

ani = FuncAnimation(fig, update, frames=steps//10, blit=True, interval=15)

plt.show()


