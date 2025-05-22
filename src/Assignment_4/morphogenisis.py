import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time

class TuringPatternModel:
    def __init__(self):
        # Grid parameters
        self.nx = 150
        self.ny = 150
        self.dx = 1.0
        self.dy = 1.0
        
        # Time parameters
        self.dt = 0.005
        self.steps = 8000
        self.plot_interval = 200
        
        # Turing pattern parameters (Gray-Scott-like system)
        self.Da = 1.0      # Activator diffusion
        self.Di = 0.5      # Inhibitor diffusion  
        self.feed = 0.037  # Feed rate (replenishes activator)
        self.kill = 0.06   # Kill rate (removes inhibitor)
        
        # Cell parameters
        self.differentiation_threshold = 0.4  # A concentration for differentiation
        self.proliferation_threshold = 0.3    # A concentration for proliferation
        self.max_cell_density = 0.8           # Maximum local cell density
        self.growth_rate = 0.002
        self.differentiation_rate = 0.01
        
        # Cell states
        self.EMPTY = 0
        self.STEM = 1
        self.DIFFERENTIATING = 2
        self.DIFFERENTIATED = 3
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the chemical concentrations and cell grid"""
        # Initialize with small random perturbations around equilibrium
        self.A = np.ones((self.nx, self.ny)) * 1.0 + np.random.normal(0, 0.01, (self.nx, self.ny))
        self.I = np.zeros((self.nx, self.ny)) + np.random.normal(0, 0.01, (self.nx, self.ny))
        
        # Add some seed perturbations to trigger pattern formation
        center_x, center_y = self.nx // 2, self.ny // 2
        for i in range(5):  # Multiple seeds for richer patterns
            x = center_x + np.random.randint(-20, 20)
            y = center_y + np.random.randint(-20, 20)
            # Create small activator depletion spots that will grow
            self.A[max(0, x-3):min(self.nx, x+4), max(0, y-3):min(self.ny, y+4)] = 0.5
            self.I[max(0, x-3):min(self.nx, x+4), max(0, y-3):min(self.ny, y+4)] = 0.25
        
        # Initialize cell grid with sparse stem cells
        self.cells = np.zeros((self.nx, self.ny), dtype=int)
        self.cell_density = np.zeros((self.nx, self.ny))  # Track local cell density
        
        # Place initial stem cells in a small central region
        for i in range(center_x-5, center_x+6):
            for j in range(center_y-5, center_y+6):
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    if np.random.random() < 0.3:  # 30% chance of initial stem cell
                        self.cells[i, j] = self.STEM
                        self.cell_density[i, j] = 0.1
    
    def laplacian(self, field):
        """Calculate 2D Laplacian using convolution"""
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
        return convolve(field, kernel, mode='wrap')
    
    def reaction_diffusion_step(self):
        """Update chemical concentrations using Gray-Scott equations"""
        # Calculate diffusion terms
        lap_A = self.laplacian(self.A)
        lap_I = self.laplacian(self.I)
        
        # Gray-Scott reaction terms
        reaction = self.A * self.I * self.I
        
        # Update equations
        dA_dt = self.Da * lap_A - reaction + self.feed * (1 - self.A)
        dI_dt = self.Di * lap_I + reaction - (self.kill + self.feed) * self.I
        
        # Apply time step
        self.A += self.dt * dA_dt
        self.I += self.dt * dI_dt
        
        # Ensure non-negative concentrations
        self.A = np.maximum(self.A, 0)
        self.I = np.maximum(self.I, 0)
    
    def update_cell_density(self):
        """Update local cell density using a smoothing kernel"""
        kernel = np.array([[0.1, 0.2, 0.1],
                          [0.2, 0.2, 0.2],
                          [0.1, 0.2, 0.1]])
        
        # Convert cell states to density contribution
        cell_contribution = np.zeros_like(self.cells, dtype=float)
        cell_contribution[self.cells == self.STEM] = 0.3
        cell_contribution[self.cells == self.DIFFERENTIATING] = 0.5
        cell_contribution[self.cells == self.DIFFERENTIATED] = 0.7
        
        self.cell_density = convolve(cell_contribution, kernel, mode='constant')
    
    def cell_dynamics_step(self):
        """Update cell states based on chemical concentrations"""
        new_cells = self.cells.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                current_state = self.cells[i, j]
                A_conc = self.A[i, j]
                I_conc = self.I[i, j]
                local_density = self.cell_density[i, j]
                
                # Calculate A/I ratio as differentiation signal
                if I_conc > 0:
                    ai_ratio = A_conc / (A_conc + I_conc)
                else:
                    ai_ratio = 1.0
                
                if current_state == self.EMPTY:
                    # Empty cells can become stem cells if conditions are right
                    if (A_conc > self.proliferation_threshold and 
                        local_density < self.max_cell_density and
                        np.random.random() < self.growth_rate):
                        new_cells[i, j] = self.STEM
                
                elif current_state == self.STEM:
                    # Stem cells can differentiate or proliferate
                    if ai_ratio < self.differentiation_threshold:
                        # Low A/I ratio triggers differentiation
                        if np.random.random() < self.differentiation_rate:
                            new_cells[i, j] = self.DIFFERENTIATING
                    
                    # Proliferation: create new stem cells in neighborhood
                    elif (A_conc > self.proliferation_threshold and 
                          local_density < self.max_cell_density and
                          np.random.random() < self.growth_rate * 0.5):
                        
                        # Find empty neighboring cells
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if (0 <= ni < self.nx and 0 <= nj < self.ny and 
                                    self.cells[ni, nj] == self.EMPTY):
                                    neighbors.append((ni, nj))
                        
                        if neighbors:
                            ni, nj = neighbors[np.random.randint(len(neighbors))]
                            new_cells[ni, nj] = self.STEM
                
                elif current_state == self.DIFFERENTIATING:
                    # Differentiating cells become fully differentiated
                    if np.random.random() < 0.1:  # 10% chance per step
                        new_cells[i, j] = self.DIFFERENTIATED
        
        self.cells = new_cells
        self.update_cell_density()
    
    def get_visualization_data(self):
        """Prepare data for visualization"""
        # Create RGB image for chemicals
        chem_img = np.zeros((self.nx, self.ny, 3))
        
        # Normalize concentrations
        A_norm = np.clip(self.A / np.max(self.A), 0, 1)
        I_norm = np.clip(self.I / np.max(self.I), 0, 1)
        
        chem_img[:, :, 1] = A_norm  # Green for activator
        chem_img[:, :, 0] = I_norm  # Red for inhibitor
        
        # Create RGB image for cells
        cell_img = np.zeros((self.nx, self.ny, 3))
        
        # Color mapping for cells
        cell_img[self.cells == self.STEM] = [0, 0, 1]        # Blue for stem cells
        cell_img[self.cells == self.DIFFERENTIATING] = [1, 1, 0]  # Yellow for differentiating
        cell_img[self.cells == self.DIFFERENTIATED] = [1, 0.5, 0] # Orange for differentiated
        
        return chem_img, cell_img
    
    def run_simulation(self):
        """Run the complete simulation"""
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        print("Starting Turing pattern simulation...")
        print("This will take a few minutes to develop clear patterns...")
        
        for step in range(self.steps):
            # Update chemistry
            self.reaction_diffusion_step()
            
            # Update cells every few chemical steps for stability
            if step % 5 == 0:
                self.cell_dynamics_step()
            
            # Plotting
            if step % self.plot_interval == 0:
                chem_img, cell_img = self.get_visualization_data()
                
                # Plot chemical concentrations
                ax1.clear()
                ax1.imshow(chem_img, origin='lower')
                ax1.set_title(f'Step {step}: Chemicals (R=Inhibitor, G=Activator)')
                ax1.axis('off')
                
                # Plot cell states
                ax2.clear()
                ax2.imshow(cell_img, origin='lower')
                ax2.set_title(f'Step {step}: Cells (Blue=Stem, Yellow=Diff, Orange=Final)')
                ax2.axis('off')
                
                # Plot activator field only
                ax3.clear()
                im3 = ax3.imshow(self.A, cmap='viridis', origin='lower')
                ax3.set_title('Activator Concentration')
                ax3.axis('off')
                
                # Plot inhibitor field only
                ax4.clear()
                im4 = ax4.imshow(self.I, cmap='plasma', origin='lower')
                ax4.set_title('Inhibitor Concentration')
                ax4.axis('off')
                
                plt.tight_layout()
                plt.pause(0.01)
                
                # Print statistics
                stem_count = np.sum(self.cells == self.STEM)
                diff_count = np.sum(self.cells == self.DIFFERENTIATING)
                final_count = np.sum(self.cells == self.DIFFERENTIATED)
                print(f"Step {step}: Stem={stem_count}, Differentiating={diff_count}, Differentiated={final_count}")
        
        plt.ioff()
        
        # Final plot
        chem_img, cell_img = self.get_visualization_data()
        ax1.imshow(chem_img, origin='lower')
        ax2.imshow(cell_img, origin='lower')
        ax3.imshow(self.A, cmap='viridis', origin='lower')
        ax4.imshow(self.I, cmap='plasma', origin='lower')
        plt.show()
        
        print("Simulation completed!")
        print(f"Final cell counts:")
        print(f"  Stem cells: {np.sum(self.cells == self.STEM)}")
        print(f"  Differentiating: {np.sum(self.cells == self.DIFFERENTIATING)}")
        print(f"  Differentiated: {np.sum(self.cells == self.DIFFERENTIATED)}")

# Run the simulation
if __name__ == "__main__":
    model = TuringPatternModel()
    model.run_simulation()