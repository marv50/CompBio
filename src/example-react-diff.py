import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm

# -----------------------------
# 1. Mesh and Time Setup
# -----------------------------
nx = 50
ny = 50
dx = dy = 1.0
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

time_step = 0.01
steps = 500
plot_interval = 50

# -----------------------------
# 2. Parameters
# -----------------------------
Da = 0.01
Di = 5.0
mu_max = 0.2
kcw = 0.05
K_IA = 1.0
theta_a = 0.02
theta_i = 0.02
Y = 0.5
Mcw = 1.0

# -----------------------------
# 3. Define Variables
# -----------------------------
A = CellVariable(name="Activator A", mesh=mesh, value=0.1)
I = CellVariable(name="Inhibitor I", mesh=mesh, value=0.05)

# Add spatial noise
A.setValue(A.value + 0.02 * np.random.random(len(A)))
I.setValue(I.value + 0.02 * np.random.random(len(I)))

# -----------------------------
# 4. Reaction Terms
# -----------------------------
def Ra(A, I):
    return (mu_max / Y) * Mcw * (A / (theta_a + A)) * (K_IA / (K_IA + I))

def Ri(I):
    return kcw * Mcw * (I / (theta_i + I))

# -----------------------------
# 5. PDEs
# -----------------------------
eq_A = TransientTerm(var=A) == DiffusionTerm(coeff=Da, var=A) - Ra(A, I)
eq_I = TransientTerm(var=I) == DiffusionTerm(coeff=Di, var=I) - Ri(I)

# -----------------------------
# 6. Combined RGB Plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

def plot_rgb():
    Amap = np.array(A.value).reshape((nx, ny))
    Imap = np.array(I.value).reshape((nx, ny))

    # Normalize to [0,1]
    Amap = (Amap - Amap.min()) / (Amap.max() - Amap.min() + 1e-8)
    Imap = (Imap - Imap.min()) / (Imap.max() - Imap.min() + 1e-8)

    rgb = np.zeros((nx, ny, 3))
    rgb[..., 1] = Amap  # Green: A
    rgb[..., 0] = Imap  # Red: I

    ax.clear()
    ax.set_title("Activator (G) + Inhibitor (R)")
    ax.imshow(rgb, origin="lower")
    ax.axis("off")
    plt.pause(0.01)

# -----------------------------
# 7. Run Simulation
# -----------------------------
for step in range(steps):
    eq_A.solve(dt=time_step)
    eq_I.solve(dt=time_step)


    print(f"Step {step}")
    plot_rgb()

plt.ioff()
plot_rgb()
plt.show()
