import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import time
import json
import os


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


@njit
def initialize_grid_njit(N, seed=42):
    np.random.seed(seed)
    a = np.ones((N, N), dtype=np.float64) * 0.5
    i = np.zeros((N, N), dtype=np.float64)

    center = N // 2
    size = N // 10
    i[center - size:center + size, center - size:center + size] = 0.25

    # Add random noise
    for x in range(N):
        for y in range(N):
            a[x, y] += np.random.uniform(-0.01, 0.01)
            i[x, y] += np.random.uniform(-0.01, 0.01)

    return a, i


@njit
def apply_boundary_conditions_njit(grid, bc_type_flag):
    # bc_type_flag: 0 for dirichlet, 1 for other types (future extension)
    if bc_type_flag == 0:  # dirichlet
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0


@njit
def compute_laplacian_njit(grid):
    N = grid.shape[0]
    laplacian = np.zeros_like(grid)

    for i in range(N):
        for j in range(N):
            # Handle periodic boundary for laplacian calculation
            up = (i - 1) % N
            down = (i + 1) % N
            left = (j - 1) % N
            right = (j + 1) % N

            laplacian[i, j] = (grid[up, j] + grid[down, j] +
                               grid[i, left] + grid[i, right] - 4 * grid[i, j])

    return laplacian


@njit
def init_cell_grid_njit(nx, ny):
    cell_grid = np.zeros((nx, ny), dtype=np.int32)
    cell_grid[nx//3, ny//2] = TRANS_CELL
    return cell_grid


@njit
def init_mass_grid_njit(nx, ny, init_mass):
    mass_grid = np.zeros((nx, ny), dtype=np.float64)
    mass_grid[nx//2, ny//2] = init_mass
    return mass_grid


@njit
def cell_growth_njit(cell_grid, mass_grid, a, i, N, mu_max, Mcw, theta_a, kcw):
    for x in range(N):
        for y in range(N):
            if cell_grid[x, y] == TRANS_CELL:
                growth = mu_max * Mcw * (a[x, y] / (theta_a + a[x, y]))
                mass_grid[x, y] += growth
                a[x, y] -= kcw * a[x, y]
    return a


@njit
def select_empty_moore_neighbors_njit(cell_grid, i, j, N):
    neighbors = np.empty((8, 2), dtype=np.int32)
    count = 0

    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N and cell_grid[ni, nj] == EMPTY_CELL:
                neighbors[count, 0] = ni
                neighbors[count, 1] = nj
                count += 1

    return neighbors[:count], count


@njit
def proliferation_njit(cell_grid, mass_grid, N, max_mass, seed_offset=0):
    np.random.seed(42 + seed_offset)
    new_cell_grid = cell_grid.copy()
    new_mass_grid = mass_grid.copy()

    for i in range(N):
        for j in range(N):
            if cell_grid[i, j] == TRANS_CELL and mass_grid[i, j] >= max_mass:
                neighbors, count = select_empty_moore_neighbors_njit(
                    cell_grid, i, j, N)
                if count > 0:
                    idx = np.random.randint(0, count)
                    ni, nj = neighbors[idx, 0], neighbors[idx, 1]
                    new_cell_grid[ni, nj] = TRANS_CELL
                    new_mass = mass_grid[i, j] / 2
                    new_mass_grid[i, j] = new_mass
                    new_mass_grid[ni, nj] = new_mass

    return new_cell_grid, new_mass_grid


@njit
def differentiation_njit(cell_grid, a, i, N):
    new_cell_grid = cell_grid.copy()
    for x in range(N):
        for y in range(N):
            if cell_grid[x, y] == TRANS_CELL and a[x, y] > 0 and i[x, y] > a[x, y]:
                new_cell_grid[x, y] = BROWN_CELL
    return new_cell_grid


@njit
def simulation_step(a, i, cell_grid, mass_grid, N, Du, Dv, f, k, dt,
                    mu_max, Mcw, theta_a, kcw, max_mass, bc_type_flag, seed_offset):
    """Single simulation step - all computations in one njit function"""

    # Reaction-diffusion step
    lap_a = compute_laplacian_njit(a)
    lap_i = compute_laplacian_njit(i)

    a_i_i = a * i * i
    a += (Du * lap_a - a_i_i + f * (1 - a)) * dt
    i += (Dv * lap_i + a_i_i - (f + k) * i) * dt

    apply_boundary_conditions_njit(a, bc_type_flag)
    apply_boundary_conditions_njit(i, bc_type_flag)

    # Cell processes
    a = cell_growth_njit(cell_grid, mass_grid, a, i,
                         N, mu_max, Mcw, theta_a, kcw)
    cell_grid, mass_grid = proliferation_njit(
        cell_grid, mass_grid, N, max_mass, seed_offset)
    cell_grid = differentiation_njit(cell_grid, a, i, N)

    return a, i, cell_grid, mass_grid


@njit
def run_simulation_batch(a, i, cell_grid, mass_grid, batch_size, N, Du, Dv, f, k, dt,
                         mu_max, Mcw, theta_a, kcw, max_mass, bc_type_flag, start_step):
    """Run a batch of simulation steps"""
    for step in range(batch_size):
        a, i, cell_grid, mass_grid = simulation_step(
            a, i, cell_grid, mass_grid, N, Du, Dv, f, k, dt,
            mu_max, Mcw, theta_a, kcw, max_mass, bc_type_flag, start_step + step
        )
    return a, i, cell_grid, mass_grid


def print_progress_bar(current, total, bar_length=50):
    """Print a progress bar"""
    progress = current / total
    block = int(bar_length * progress)
    bar = "█" * block + "░" * (bar_length - block)
    percent = progress * 100
    print(
        f"\rProgress: [{bar}] {percent:.1f}% ({current}/{total})", end="", flush=True)


def get_cell_rgb(cell_grid, cell_colors):
    """Convert cell grid to RGB array using config-defined colors."""
    rgb_array = np.zeros((cell_grid.shape[0], cell_grid.shape[1], 3))
    empty_rgb = np.array(cell_colors["empty"]) / 255.0
    trans_rgb = np.array(cell_colors["transitional"]) / 255.0
    brown_rgb = np.array(cell_colors["brown"]) / 255.0

    for i in range(cell_grid.shape[0]):
        for j in range(cell_grid.shape[1]):
            if cell_grid[i, j] == EMPTY_CELL:
                rgb_array[i, j] = empty_rgb
            elif cell_grid[i, j] == TRANS_CELL:
                rgb_array[i, j] = trans_rgb
            elif cell_grid[i, j] == BROWN_CELL:
                rgb_array[i, j] = brown_rgb
    return rgb_array


def run_full_simulation(params):
    N = params['N']
    Du = params['Du']
    Dv = params['Dv']
    f = params['f']
    k = params['k']
    dt = params['dt']
    total_steps = params['total_steps']
    save_interval = params['save_interval']
    mu_max = params['mu_max']
    Mcw = params['Mcw']
    theta_a = params['theta_a']
    kcw = params['kcw']
    init_mass = params['init_mass']
    max_mass = params['max_mass']
    bc_type_flag = 0 if params['bc_type'] == "dirichlet" else 1

    # Initialize
    a, i = initialize_grid_njit(N)
    cell_grid = init_cell_grid_njit(N, N)
    mass_grid = init_mass_grid_njit(N, N, init_mass)

    # Pre-compile
    a_test, i_test = initialize_grid_njit(10)
    cell_test = init_cell_grid_njit(10, 10)
    mass_test = init_mass_grid_njit(10, 10, init_mass)
    run_simulation_batch(a_test, i_test, cell_test, mass_test, 1, 10, Du, Dv, f, k, dt,
                         mu_max, Mcw, theta_a, kcw, max_mass, bc_type_flag, 0)

    # Storage for results - now including activator (a)
    num_frames = total_steps // save_interval
    saved_a = np.zeros((num_frames, N, N))  # Added for activator storage
    saved_i = np.zeros((num_frames, N, N))
    saved_cells = np.zeros((num_frames, N, N), dtype=np.int32)
    saved_times = np.zeros(num_frames)

    print(f"\nRunning simulation for {total_steps} steps...")
    frame_idx = 0
    batch_size = save_interval
    for step in range(0, total_steps, batch_size):
        a, i, cell_grid, mass_grid = run_simulation_batch(
            a, i, cell_grid, mass_grid, batch_size, N, Du, Dv, f, k, dt,
            mu_max, Mcw, theta_a, kcw, max_mass, bc_type_flag, step
        )

        if frame_idx < num_frames:
            saved_a[frame_idx] = a.copy()  # Save activator values
            saved_i[frame_idx] = i.copy()
            saved_cells[frame_idx] = cell_grid.copy()
            saved_times[frame_idx] = step + batch_size
            frame_idx += 1

        print_progress_bar(step + batch_size, total_steps)

    print("\nSimulation complete.")
    return saved_a, saved_i, saved_cells, saved_times


def show_animation(saved_a, saved_i, saved_cells, saved_times, save_dir, save_plots=False):
    """Display the animation using saved simulation data"""
    print("\nPreparing animation...")
    if save_plots:
        save_all(saved_a, saved_cells, saved_times, save_dir)
        print(f"Plots saved to '{save_dir}'")

    # Setup figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    # Initial plots
    im1 = ax1.imshow(saved_i[0], cmap='inferno', interpolation='bilinear',
                     vmin=saved_i.min(), vmax=saved_i.max())
    ax1.set_title("Inhibitor Concentration (i)")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(get_cell_rgb(
        saved_cells[0], simulation_config["cell_colors"]), interpolation='nearest')
    ax2.set_title("Cell States")

    # Add legend for cell types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Empty'),
        Patch(facecolor=(0.4, 0.8, 1.0), label='Transitional'),
        Patch(facecolor=(0.5, 0.3, 0.1), label='Brown')
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(1.0, 1.0))

    # Animation update function
    def update(frame):
        im1.set_array(saved_i[frame])
        im2.set_array(get_cell_rgb(
            saved_cells[frame], simulation_config["cell_colors"]))

        # Update titles with time information
        ax1.set_title(
            f"Inhibitor Concentration - Step: {int(saved_times[frame])}")
        ax2.set_title(f"Cell States - Step: {int(saved_times[frame])}")

        return [im1, im2]

    print("Starting animation...")
    ani = FuncAnimation(fig, update, frames=len(saved_times),
                        blit=True, interval=100, repeat=True)

    plt.show()
    return ani


def plot_total_activator_over_time(saved_a, saved_times):
    """Plot total activator over time."""
    total_a = np.sum(saved_a, axis=(1, 2))  # Sum across spatial dimensions
    
    fig, ax = plt.subplots()
    ax.plot(saved_times, total_a, 'g-', linewidth=2, label='Total Activator')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total Activator (a)")
    ax.set_title("Total Activator Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_phase_diagram_off_vs_on(saved_cells):
    """Create a phase diagram showing off versus on cells over time."""
    # Count different cell types at each time step
    off_cells = np.sum(saved_cells == EMPTY_CELL, axis=(1, 2))  # Empty cells = "off"
    on_cells = np.sum((saved_cells == TRANS_CELL) | (saved_cells == BROWN_CELL), axis=(1, 2))  # Active cells = "on"
    
    fig, ax = plt.subplots()
    ax.plot(off_cells, on_cells, 'bo-', markersize=4, linewidth=1.5)
    ax.set_xlabel("Off Cells (Empty)")
    ax.set_ylabel("On Cells (Transitional + Brown)")
    ax.set_title("Phase Plot: Off vs On Cells")
    ax.grid(True, alpha=0.3)
    
    # Add arrow to show direction of time evolution
    if len(off_cells) > 1:
        # Add arrows between some points to show time direction
        for i in range(0, len(off_cells)-1, max(1, len(off_cells)//10)):
            if i+1 < len(off_cells):
                ax.annotate('', xy=(off_cells[i+1], on_cells[i+1]), 
                           xytext=(off_cells[i], on_cells[i]),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    return fig


def plot_cell_counts_over_time(saved_cells, saved_times):
    """Plot the number of switched (brown) and non-switched (transitional) cells over time."""
    trans_counts = np.sum(saved_cells == TRANS_CELL, axis=(1, 2))
    brown_counts = np.sum(saved_cells == BROWN_CELL, axis=(1, 2))

    fig, ax = plt.subplots()
    ax.plot(saved_times, trans_counts,
            label='Transitional Cells', color='blue')
    ax.plot(saved_times, brown_counts, label='Brown Cells', color='brown')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cell Count")
    ax.set_title("Cell States Over Time")
    ax.legend()
    return fig


def save_all(saved_a, saved_cells, saved_times, save_dir):
    """Save all analysis plots to the specified directory with a base name and plot-specific suffixes as PDFs."""
    os.makedirs(save_dir, exist_ok=True)

    plot_info = [
        (plot_total_activator_over_time(saved_a, saved_times), "total_activator_plot"),
        (plot_phase_diagram_off_vs_on(saved_cells), "phase_off_on_plot"),
        (plot_cell_counts_over_time(saved_cells, saved_times), "cell_count_plot")
    ]

    base_filename = input(
        "Enter a base filename (without extension) for your plots: ").strip()
    if not base_filename:
        print("No filename provided. Skipping plot saving.")
        for fig, _ in plot_info:
            plt.close(fig)
        return

    for fig, plot_suffix in plot_info:
        full_filename = f"{plot_suffix}_{base_filename}.pdf"
        filepath = os.path.join(save_dir, full_filename)
        fig.savefig(filepath, format='pdf')
        plt.close(fig)
        print(f"Saved '{plot_suffix}' as {filepath}")


# Main execution
if __name__ == "__main__":
    # Load config
    config_path = "src/Assignment_4/configs/config_2.json"
    simulation_config = load_config(config_path)

    # Constants for cell states
    EMPTY_CELL = 0
    TRANS_CELL = 1
    BROWN_CELL = 2

    # Run simulation
    saved_a, saved_i, saved_cells, saved_times = run_full_simulation(simulation_config)

    save_dir = r"src\Assignment_4\fig"

    # Show animation and optionally save plots
    animation = show_animation(saved_a, saved_i, saved_cells, saved_times, save_dir,
                               save_plots=False)