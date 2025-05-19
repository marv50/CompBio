import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import time

# --- Function Definitions ---

def initialize_fields(nx, ny, Lx, Ly, pattern_type='spot'):
    """
    Initialize the fields u and v on a 2D grid.
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid points in x and y directions
    Lx, Ly : float
        Domain size in x and y directions
    pattern_type : str
        Type of initial condition ('spot', 'random', or 'stripe')
    
    Returns:
    --------
    u, v : 2D arrays
        Initial concentrations
    x, y : 2D arrays
        Spatial grid
    """
    # Check grid dimensions
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be at least 3x3")
    
    # Create 2D grid
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Initialize fields
    u = np.ones((ny, nx))
    v = np.zeros((ny, nx))
    
    if pattern_type == 'spot':
        # Create a spot in the center
        center_x, center_y = Lx/2, Ly/2
        radius = min(Lx, Ly) * 0.1
        mask = ((xx - center_x)**2 + (yy - center_y)**2) < radius**2
        u[mask] = 0.5
        v[mask] = 0.25
        
    elif pattern_type == 'random':
        # Random perturbations
        u = u + 0.05 * (np.random.random((ny, nx)) - 0.5)
        v = v + 0.05 * (np.random.random((ny, nx)) - 0.5)
        
    elif pattern_type == 'stripe':
        # Create a vertical stripe
        stripe_width = Lx * 0.1
        mask = np.abs(xx - Lx/2) < stripe_width
        u[mask] = 0.5
        v[mask] = 0.25
    
    return u, v, xx, yy

def compute_laplacian_2d(field, dx, dy):
    """
    Compute the 2D Laplacian using central differences and periodic boundary conditions.
    Uses a more robust NumPy-based approach to handle boundaries.
    
    Parameters:
    -----------
    field : 2D array
        The field for which to compute the Laplacian
    dx, dy : float
        Grid spacing in x and y directions
    
    Returns:
    --------
    laplacian : 2D array
        The Laplacian of the field
    """
    ny, nx = field.shape
    
    # Create shifted arrays for periodic boundary conditions
    # For x direction
    field_x_plus = np.roll(field, -1, axis=1)
    field_x_minus = np.roll(field, 1, axis=1)
    
    # For y direction
    field_y_plus = np.roll(field, -1, axis=0)
    field_y_minus = np.roll(field, 1, axis=0)
    
    # Compute Laplacian with periodic boundary conditions
    laplacian = (
        (field_x_plus - 2*field + field_x_minus) / (dx**2) +
        (field_y_plus - 2*field + field_y_minus) / (dy**2)
    )
    
    return laplacian

def update_gray_scott(u, v, Du, Dv, F, k, dt, dx, dy):
    """
    Update the fields u and v using the Gray-Scott reaction-diffusion model.
    
    Parameters:
    -----------
    u, v : 2D arrays
        Current concentrations
    Du, Dv : float
        Diffusion coefficients
    F : float
        Feed rate
    k : float
        Kill rate
    dt : float
        Time step
    dx, dy : float
        Grid spacing
    
    Returns:
    --------
    u_new, v_new : 2D arrays
        Updated concentrations
    """
    # Compute Laplacians
    lap_u = compute_laplacian_2d(u, dx, dy)
    lap_v = compute_laplacian_2d(v, dx, dy)
    
    # Gray-Scott reaction terms
    reaction_u = -u * v**2 + F * (1 - u)
    reaction_v = u * v**2 - (F + k) * v
    
    # Update fields
    u_new = u + dt * (Du * lap_u + reaction_u)
    v_new = v + dt * (Dv * lap_v + reaction_v)
    
    # Enforce bounds to prevent instabilities
    u_new = np.clip(u_new, 0.0, 1.0)
    v_new = np.clip(v_new, 0.0, 1.0)
    
    return u_new, v_new

def compute_stability_dt(dx, dy, Du, Dv):
    """
    Compute a stable time step based on the CFL condition.
    
    Parameters:
    -----------
    dx, dy : float
        Grid spacing
    Du, Dv : float
        Diffusion coefficients
    
    Returns:
    --------
    dt : float
        Stable time step
    """
    # For diffusion equation, stability requires dt <= 0.5 * min(dx², dy²) / max(Du, Dv)
    dt_max = 0.5 * min(dx**2, dy**2) / max(Du, Dv)
    
    # Apply a safety factor
    safety_factor = 0.9
    dt = safety_factor * dt_max
    
    return dt

def simulate_gray_scott(nx, ny, Lx, Ly, Du, Dv, F, k, dt=None, T=1000, pattern_type='spot', max_frames=100):
    """
    Run the Gray-Scott reaction-diffusion simulation.
    
    Parameters:
    -----------
    nx, ny : int
        Number of grid points in x and y directions
    Lx, Ly : float
        Domain size
    Du, Dv : float
        Diffusion coefficients
    F : float
        Feed rate
    k : float
        Kill rate
    dt : float or None
        Time step (if None, will be calculated automatically)
    T : float
        Total simulation time
    pattern_type : str, optional
        Type of initial condition
    max_frames : int, optional
        Maximum number of frames to save
    
    Returns:
    --------
    u_history, v_history : list of 2D arrays
        Time series of concentrations
    """
    # Initialize fields
    u, v, xx, yy = initialize_fields(nx, ny, Lx, Ly, pattern_type)
    
    # Calculate grid spacing
    if nx <= 1 or ny <= 1:
        raise ValueError("Grid dimensions must be > 1")
    
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Calculate stable time step if not provided
    if dt is None:
        dt = compute_stability_dt(dx, dy, Du, Dv)
        print(f"Using stable time step: dt = {dt:.8f}")
    else:
        # Check if provided dt is stable
        dt_stable = compute_stability_dt(dx, dy, Du, Dv)
        if dt > dt_stable:
            print(f"WARNING: Provided dt={dt} exceeds stability limit {dt_stable:.8f}")
            print(f"This may cause numerical instability.")
    
    # Run simulation
    time_steps = int(T / dt)
    save_steps = max(1, time_steps // max_frames)  # Save at most max_frames
    
    u_history = [u.copy()]
    v_history = [v.copy()]
    times = [0]
    
    start_time = time.time()
    print(f"Starting simulation: {time_steps} steps")
    
    for t in range(time_steps):
        # Update fields
        u, v = update_gray_scott(u, v, Du, Dv, F, k, dt, dx, dy)
        
        # Save fields periodically
        if (t+1) % save_steps == 0:
            u_history.append(u.copy())
            v_history.append(v.copy())
            times.append((t+1) * dt)
            
            # Print progress every 10%
            if (t+1) % (time_steps // 10) == 0:
                progress = (t+1) / time_steps * 100
                elapsed = time.time() - start_time
                print(f"Progress: {progress:.1f}%, Time elapsed: {elapsed:.1f}s")
    
    print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
    return u_history, v_history, times, xx, yy

# --- Parameter sets for interesting patterns ---
pattern_params = {
    'spots': {'F': 0.035, 'k': 0.065},          # Small spots
    'worms': {'F': 0.034, 'k': 0.063},          # Worm-like structures
    'maze': {'F': 0.029, 'k': 0.057},           # Maze-like pattern
    'waves': {'F': 0.018, 'k': 0.051},          # Propagating waves
    'fingerprint': {'F': 0.026, 'k': 0.055},    # Fingerprint-like pattern
}

# --- Run the Simulation ---
def run_simulation(pattern_name='worms', nx=200, ny=200, T=2000, auto_dt=True):
    """
    Run a Gray-Scott simulation with the specified pattern and parameters
    
    Parameters:
    -----------
    pattern_name : str
        Name of pattern to simulate (from pattern_params)
    nx, ny : int
        Grid dimensions
    T : float
        Total simulation time
    auto_dt : bool
        Whether to automatically calculate stable time step
    """
    # Check if pattern exists
    if pattern_name not in pattern_params:
        valid_patterns = list(pattern_params.keys())
        raise ValueError(f"Unknown pattern '{pattern_name}'. Valid options: {valid_patterns}")
    
    params = pattern_params[pattern_name]
    
    # Simulation parameters
    Lx, Ly = 2.0, 2.0       # Domain size
    Du = 0.2                # Diffusion coefficient for u (slower)
    Dv = 0.1                # Diffusion coefficient for v (faster)
    F = params['F']         # Feed rate
    k = params['k']         # Kill rate
    
    dt = None if auto_dt else 0.001  # Automatically calculated if None
    
    print(f"Running Gray-Scott simulation with pattern: {pattern_name}")
    print(f"Parameters: F={F}, k={k}, Grid: {nx}x{ny}")
    
    # Run simulation
    u_history, v_history, times, xx, yy = simulate_gray_scott(
        nx, ny, Lx, Ly, Du, Dv, F, k, dt, T, pattern_type='random'
    )
    
    # Plot final u field
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, u_history[-1], cmap='viridis')
    plt.colorbar(label='u concentration')
    plt.title(f"Gray-Scott Pattern: {pattern_name} (t={times[-1]:.1f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"gray_scott_{pattern_name}_final.png")
    plt.show()
    
    # Create animation (efficient approach)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Only create one imshow object and update data
    img = ax.imshow(u_history[0], cmap='viridis', 
                    extent=[0, Lx, 0, Ly], origin='lower',
                    animated=True, interpolation='nearest')
    cbar = fig.colorbar(img, ax=ax, label='u concentration')
    title = ax.set_title(f"Gray-Scott Pattern: {pattern_name} (t=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    def update_frame(i):
        img.set_array(u_history[i])
        title.set_text(f"Gray-Scott Pattern: {pattern_name} (t={times[i]:.1f})")
        return [img]  # Return list of artists that were modified
    
    ani = FuncAnimation(fig, update_frame, frames=len(u_history), 
                        interval=100, blit=True)
    plt.tight_layout()
    
    # Save animation (optional)
    # ani.save(f"gray_scott_{pattern_name}.mp4", writer='ffmpeg', fps=10)
    
    plt.show()
    
    return u_history, v_history, times, xx, yy

# --- Experiment with different patterns ---
def run_pattern_experiment():
    """Run simulations for all predefined patterns and display results"""
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, (name, params) in enumerate(pattern_params.items()):
        if i >= len(axs):
            break
            
        print(f"Simulating {name} pattern...")
        
        # Run short simulation
        nx, ny = 100, 100  # Smaller grid for quicker runs
        Lx, Ly = 2.0, 2.0
        Du, Dv = 0.2, 0.1
        F, k = params['F'], params['k']
        
        # Calculate stable dt
        dx, dy = Lx/(nx-1), Ly/(ny-1)
        dt = compute_stability_dt(dx, dy, Du, Dv)
        
        # Shorter simulation time for the experiment
        T = 500
        
        u_hist, v_hist, times, xx, yy = simulate_gray_scott(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly,
            Du=Du, Dv=Dv, F=F, k=k,
            dt=dt, T=T, pattern_type='random'
        )
        
        # Plot final state
        im = axs[i].imshow(u_hist[-1], cmap='viridis', origin='lower')
        axs[i].set_title(f"{name}\nF={params['F']}, k={params['k']}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("gray_scott_patterns.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Run either a single simulation or the pattern experiment
    run_simulation(pattern_name='worms', nx=200, ny=200, T=2, auto_dt=True)
    #run_pattern_experiment()