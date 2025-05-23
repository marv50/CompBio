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
    print("\nPreparing animation...")
    
    # Get filename once if saving plots
    base_filename = None
    if save_plots:
        base_filename = input("Enter a base filename (without extension) for your plots: ").strip()
        if not base_filename:
            print("No filename provided. Skipping plot saving.")
            save_plots = False
        else:
            save_all(saved_a, saved_cells, saved_times, save_dir, base_filename)
            print(f"Plots saved to '{save_dir}'")

    # Create figure with improved formatting
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Inhibitor plot with better formatting
    im1 = ax1.imshow(saved_i[0], cmap='inferno', interpolation='bilinear',
                     vmin=saved_i.min(), vmax=saved_i.max())
    ax1.set_title("Inhibitor Concentration (i)", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Concentration", fontsize=11)

    # Cell states plot with better formatting
    im2 = ax2.imshow(get_cell_rgb(saved_cells[0], simulation_config["cell_colors"]), interpolation='nearest')
    ax2.set_title("Cell States", fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel("X Position", fontsize=12)
    ax2.set_ylabel("Y Position", fontsize=12)

    # Activator plot with better formatting
    im3 = ax3.imshow(saved_a[0], cmap='viridis', interpolation='bilinear',
                     vmin=saved_a.min(), vmax=saved_a.max())
    ax3.set_title("Activator Concentration (a)", fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel("X Position", fontsize=12)
    ax3.set_ylabel("Y Position", fontsize=12)
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("Concentration", fontsize=11)

    # Improved legend for cell states
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(simulation_config["cell_colors"]["empty"])/255.0, 
              edgecolor='black', label='Empty', linewidth=1),
        Patch(facecolor=np.array(simulation_config["cell_colors"]["transitional"])/255.0, 
              edgecolor='black', label='Transitional', linewidth=1),
        Patch(facecolor=np.array(simulation_config["cell_colors"]["brown"])/255.0, 
              edgecolor='black', label='Differentiated', linewidth=1)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
               frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Add grid and improve aesthetics
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(labelsize=10)
        ax.set_aspect('equal')

    def update(frame):
        im1.set_array(saved_i[frame])
        im2.set_array(get_cell_rgb(saved_cells[frame], simulation_config["cell_colors"]))
        im3.set_array(saved_a[frame])

        step = int(saved_times[frame])
        ax1.set_title(f"Inhibitor (i) - Step: {step}", fontsize=14, fontweight='bold', pad=20)
        ax2.set_title(f"Cell States - Step: {step}", fontsize=14, fontweight='bold', pad=20)
        ax3.set_title(f"Activator (a) - Step: {step}", fontsize=14, fontweight='bold', pad=20)

        return [im1, im2, im3]

    print("Starting animation...")
    ani = FuncAnimation(fig, update, frames=len(saved_times),
                        blit=True, interval=100, repeat=True)

    # Save animation if requested
    if save_plots and base_filename:
        # Try to save as MP4 first, fallback to GIF if ffmpeg not available
        try:
            animation_path = os.path.join(save_dir, f"animation_{base_filename}.mp4")
            print(f"Saving animation as MP4 to {animation_path}...")
            ani.save(animation_path, writer='ffmpeg', fps=10, bitrate=1800)
            print(f"Animation saved as {animation_path}")
        except Exception as e:
            print(f"Failed to save as MP4: {e}")
            print("Trying to save as GIF instead...")
            try:
                animation_path = os.path.join(save_dir, f"animation_{base_filename}.gif")
                ani.save(animation_path, writer='pillow', fps=5)
                print(f"Animation saved as GIF: {animation_path}")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")
                print("Animation saving failed. Please check if ffmpeg or pillow is properly installed.")

    plt.show()
    return ani


def plot_total_activator_over_time(saved_a, saved_times):
    """Plot total activator over time."""
    total_a = np.sum(saved_a, axis=(1, 2))  # Sum across spatial dimensions
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(saved_times, total_a, 'g-', linewidth=2.5, label='Total Activator')
    ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Activator (a)", fontsize=12, fontweight='bold')
    ax.set_title("Total Activator Over Time", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_phase_diagram_differentiated_vs_undifferentiated(saved_cells):
    """Create a simple phase diagram showing differentiated versus undifferentiated cells over time."""
    # Undifferentiated cells: Empty + Transitional
    undifferentiated_cells = np.sum((saved_cells == EMPTY_CELL) | (saved_cells == TRANS_CELL), axis=(1, 2))
    # Differentiated cells: Brown cells
    differentiated_cells = np.sum(saved_cells == BROWN_CELL, axis=(1, 2))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(undifferentiated_cells, differentiated_cells, 
                        c=range(len(undifferentiated_cells)), cmap='plasma', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Undifferentiated Cells (Empty + Transitional)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Differentiated Cells (Brown)", fontsize=12, fontweight='bold')
    ax.set_title("Phase Diagram: Differentiated vs Undifferentiated Cells", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Add arrows to show time direction
    if len(undifferentiated_cells) > 1:
        for i in range(0, len(undifferentiated_cells)-1, max(1, len(undifferentiated_cells)//8)):
            if i+1 < len(undifferentiated_cells):
                ax.annotate('', xy=(undifferentiated_cells[i+1], differentiated_cells[i+1]), 
                           xytext=(undifferentiated_cells[i], differentiated_cells[i]),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1.5))
    
    plt.tight_layout()
    return fig


def plot_cell_counts_over_time(saved_cells, saved_times):
    """Plot the number of switched (brown) and non-switched (transitional) cells over time."""
    trans_counts = np.sum(saved_cells == TRANS_CELL, axis=(1, 2))
    brown_counts = np.sum(saved_cells == BROWN_CELL, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(saved_times, trans_counts, label='Transitional Cells', 
            color='dodgerblue', linewidth=2.5, marker='o', markersize=4)
    ax.plot(saved_times, brown_counts, label='Differentiated Cells', 
            color='saddlebrown', linewidth=2.5, marker='s', markersize=4)
    ax.set_xlabel("Time Step", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cell Count", fontsize=12, fontweight='bold')
    ax.set_title("Cell States Over Time", fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def plot_time_progression_snapshots(saved_a, saved_i, saved_cells, saved_times):
    """Create a static plot showing snapshots of the animation at different time points."""
    # Select 3 time points evenly distributed
    n_snapshots = 3
    indices = np.linspace(0, len(saved_times)-1, n_snapshots, dtype=int)
    
    fig, axes = plt.subplots(3, n_snapshots, figsize=(4*n_snapshots, 12))
    
    for i, idx in enumerate(indices):
        # Inhibitor
        im1 = axes[0, i].imshow(saved_i[idx], cmap='inferno', interpolation='bilinear')
        axes[0, i].set_title(f'Inhibitor - Step {int(saved_times[idx])}', fontsize=12, fontweight='bold')
        axes[0, i].set_xlabel('X', fontsize=10)
        axes[0, i].set_ylabel('Y', fontsize=10)
        axes[0, i].tick_params(labelsize=8)
        
        # Cell states
        axes[1, i].imshow(get_cell_rgb(saved_cells[idx], simulation_config["cell_colors"]), interpolation='nearest')
        axes[1, i].set_title(f'Cells - Step {int(saved_times[idx])}', fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel('X', fontsize=10)
        axes[1, i].set_ylabel('Y', fontsize=10)
        axes[1, i].tick_params(labelsize=8)
        
        # Activator
        im3 = axes[2, i].imshow(saved_a[idx], cmap='viridis', interpolation='bilinear')
        axes[2, i].set_title(f'Activator - Step {int(saved_times[idx])}', fontsize=12, fontweight='bold')
        axes[2, i].set_xlabel('X', fontsize=10)
        axes[2, i].set_ylabel('Y', fontsize=10)
        axes[2, i].tick_params(labelsize=8)
        
        # Set equal aspect ratio for all subplots
        for j in range(3):
            axes[j, i].set_aspect('equal')
    
    # Add colorbars for the first and last columns
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0], shrink=0.6, label='Inhibitor')
    plt.colorbar(axes[2, 0].images[0], ax=axes[2, 0], shrink=0.6, label='Activator')
    
    plt.suptitle('Simulation Time Progression', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def save_all(saved_a, saved_cells, saved_times, save_dir, base_filename):
    """Save all analysis plots to the specified directory with a base name and plot-specific suffixes as PDFs."""
    os.makedirs(save_dir, exist_ok=True)

    plot_info = [
        (plot_total_activator_over_time(saved_a, saved_times), "total_activator_plot"),
        (plot_phase_diagram_differentiated_vs_undifferentiated(saved_cells), "phase_differentiated_plot"),
        (plot_cell_counts_over_time(saved_cells, saved_times), "cell_count_plot"),
        (plot_time_progression_snapshots(saved_a, saved_i, saved_cells, saved_times), "time_progression_plot")
    ]

    for fig, plot_suffix in plot_info:
        full_filename = f"{plot_suffix}_{base_filename}.pdf"
        filepath = os.path.join(save_dir, full_filename)
        fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved '{plot_suffix}' as {filepath}")


# Main execution
if __name__ == "__main__":
    # Load config
    config_path = "src/Assignment_4/configs/deer.json"
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