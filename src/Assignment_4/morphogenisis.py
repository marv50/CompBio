import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm

# -----------------------------
# 1. Configuration and Parameters
# -----------------------------


def get_config():
    return {
        "nx": 100,
        "ny": 100,
        "dx": 1.0,
        "dy": 1.0,
        "time_step": 0.01,
        "steps": 1000,
        "plot_interval": 1,

        # Reaction-diffusion pattern settings
        "Da": 0.5,   
        "Di": 0.5,      
        "mu_max": 0.08,  
        "kcw": 0.01,    
        "K_IA": 0.3,    
        "theta_a": 0.02,
        "theta_i": 0.01,
        "Y": 0.5,
        "Mcw": 0.5,

        # Cell parameters
        "init_mass": 0.05,
        "max_mass": 0.12,
    }


# -----------------------------
# 2. Setup Functions
# -----------------------------


def create_mesh(config):
    return Grid2D(dx=config["dx"], dy=config["dy"], nx=config["nx"], ny=config["ny"])


def initialize_variables(mesh, config):
    nx, ny = config["nx"], config["ny"]
    A = CellVariable(name="Activator A", mesh=mesh, value=0.2)
    I = CellVariable(name="Inhibitor I", mesh=mesh, value=0.2)

    A.setValue(A.value + 0.02 * np.random.random(len(A)))
    I.setValue(I.value + 0.02 * np.random.random(len(I)))

    # Create an activator well around the center
    A_array = A.value.copy().reshape((nx, ny))
    cx, cy = nx // 2, ny // 2
    A_array[cx, cy] += 0.1

    A.setValue(A_array.flatten())

    return A, I


# -----------------------------
# 3. Reaction Functions
# -----------------------------


def Ra(A, I, config):
    return (config["mu_max"] / config["Y"]) * config["Mcw"] * (A / (config["theta_a"] + A)) * \
           (config["K_IA"] / (config["K_IA"] + I))


def Ri(I, config):
    return config["kcw"] * config["Mcw"] * (I / (config["theta_i"] + I))

# -----------------------------
# 4. PDE Equations
# -----------------------------


def create_equations(A, I, config):
    eq_A = TransientTerm(var=A) == DiffusionTerm(
        coeff=config["Da"], var=A) - Ra(A, I, config)
    eq_I = TransientTerm(var=I) == DiffusionTerm(
        coeff=config["Di"], var=I) - Ri(I, config)
    return eq_A, eq_I

# -----------------------------
# 5. Cell Dynamics
# -----------------------------


# Cell states
EMPTY_CELL = 0
TRANS_CELL = 1  # Transit-amplifying cell
BROWN_CELL = 2  # Differentiated cell


def init_cell_grid(nx, ny):
    cell_grid = np.zeros((nx, ny))
    cell_grid[nx//2, ny//2] = TRANS_CELL  # Initial cell at the center
    return cell_grid


def init_mass_grid(nx, ny, init_mass):
    mass_grid = np.zeros((nx, ny))
    mass_grid[nx//2, ny//2] = init_mass  # Initial mass at the center
    return mass_grid


def read_concentration(A, I, config):
    nx, ny = config["nx"], config["ny"]
    A_conc = np.array(A.value).reshape((nx, ny))
    I_conc = np.array(I.value).reshape((nx, ny))
    return A_conc, I_conc


def cell_growth(cell_grid, mass_grid, A_conc, I_conc, config):
    """Update cell mass based on activator concentration"""
    nx, ny = config["nx"], config["ny"]

    for i in range(nx):
        for j in range(ny):
            if cell_grid[i, j] == TRANS_CELL:
                # Growth based on activator concentration
                growth = config["mu_max"] * config["Mcw"] * \
                    (A_conc[i, j] / (config["theta_a"] + A_conc[i, j]))
                mass_grid[i, j] += growth
                # Cells consume activator for growth
                A_conc[i, j] -= config["kcw"] * A_conc[i, j]

    return A_conc


def select_empty_moore_neighbors(cell_grid, i, j, nx, ny):
    """Find empty neighboring cells"""
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                if cell_grid[ni, nj] == EMPTY_CELL:
                    neighbors.append((ni, nj))
    return neighbors


def proliferation(cell_grid, mass_grid, config):
    """Handle cell division when mass exceeds threshold"""
    nx, ny = config["nx"], config["ny"]

    # Create a copy to avoid changes affecting other cells during iteration
    new_cell_grid = cell_grid.copy()
    new_mass_grid = mass_grid.copy()

    for i in range(nx):
        for j in range(ny):
            if cell_grid[i, j] == TRANS_CELL and mass_grid[i, j] >= config["max_mass"]:
                neighbors = select_empty_moore_neighbors(
                    cell_grid, i, j, nx, ny)
                if neighbors:
                    # Choose random empty neighbor cell for division
                    ni, nj = neighbors[np.random.randint(len(neighbors))]
                    new_cell_grid[ni, nj] = TRANS_CELL

                    # Divide mass between parent and daughter cell
                    new_mass_grid[i, j] = mass_grid[i, j] / 2
                    new_mass_grid[ni, nj] = mass_grid[i, j] / 2

    return new_cell_grid, new_mass_grid


def differentiation(cell_grid, A_conc, I_conc, config):
    """Handle cell differentiation based on A/I ratio"""
    nx, ny = config["nx"], config["ny"]

    new_cell_grid = cell_grid.copy()

    for i in range(nx):
        for j in range(ny):
            if cell_grid[i, j] == TRANS_CELL:
                # Avoid division by zero
                if A_conc[i, j] > 0:
                    if I_conc[i, j] > A_conc[i, j]:
                        # Inhibitor concentration is higher than activator
                        new_cell_grid[i, j] = BROWN_CELL

    return new_cell_grid

# -----------------------------
# 6. Plotting
# -----------------------------


def plot_simulation(A, I, cell_grid, config, step, ax1, ax2):
    """Plot both the chemical concentrations and cell states"""
    nx, ny = config["nx"], config["ny"]

    # Plot chemical concentrations (A and I)
    Amap = np.array(A.value).reshape((nx, ny))
    Imap = np.array(I.value).reshape((nx, ny))

    Amap_norm = (Amap - Amap.min()) / (Amap.max() - Amap.min() + 1e-8)
    Imap_norm = (Imap - Imap.min()) / (Imap.max() - Imap.min() + 1e-8)

    rgb = np.zeros((nx, ny, 3))
    rgb[..., 1] = Amap_norm  # Green channel for activator
    rgb[..., 0] = Imap_norm  # Red channel for inhibitor

    ax1.clear()
    ax1.set_title(f"Step {step}: Activator (G) + Inhibitor (R)")
    ax1.imshow(rgb, origin="lower")
    ax1.axis("off")

    # Plot cell states
    cell_colors = np.zeros((nx, ny, 3))
    # Empty cells: black
    # Transit cells: blue
    cell_colors[cell_grid == TRANS_CELL, 2] = 1.0
    # Differentiated cells: brown
    brown_mask = cell_grid == BROWN_CELL
    cell_colors[brown_mask, 0] = 0.6  # Red component
    cell_colors[brown_mask, 1] = 0.3  # Green component

    ax2.clear()
    ax2.set_title(f"Step {step}: Cell States")
    ax2.imshow(cell_colors, origin="lower")
    ax2.axis("off")

    plt.tight_layout()
    plt.pause(0.01)

# -----------------------------
# 7. Main Simulation
# -----------------------------


def run_simulation():
    config = get_config()
    mesh = create_mesh(config)
    A, I = initialize_variables(mesh, config)
    eq_A, eq_I = create_equations(A, I, config)

    # Initialize cell grid and mass grid
    cell_grid = init_cell_grid(config["nx"], config["ny"])
    mass_grid = init_mass_grid(config["nx"], config["ny"], config["init_mass"])

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for step in range(config["steps"]):
        # Solve chemical diffusion-reaction equations
        eq_A.solve(dt=config["time_step"])
        eq_I.solve(dt=config["time_step"])

        # Update cell dynamics
        A_conc, I_conc = read_concentration(A, I, config)

        # Cell growth
        A_conc = cell_growth(cell_grid, mass_grid, A_conc, I_conc, config)
        # Update the A CellVariable with modified concentrations
        A.setValue(A_conc.flatten())

        # Cell differentiation
        cell_grid = differentiation(cell_grid, A_conc, I_conc, config)

        # Cell proliferation
        cell_grid, mass_grid = proliferation(cell_grid, mass_grid, config)

        # Plotting
        if step % config["plot_interval"] == 0:
            print(f"Step {step}")
            plot_simulation(A, I, cell_grid, config, step, ax1, ax2)

    plt.ioff()
    plot_simulation(A, I, cell_grid, config, config["steps"], ax1, ax2)
    plt.show()


# -----------------------------
# 8. Entry Point
# -----------------------------
if __name__ == "__main__":
    run_simulation()
