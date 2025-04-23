import numpy as np
import matplotlib.pyplot as plt

def simulate_sde_system(
    c, a, b, beta, gamma,
    sigma1, sigma2,
    U0, S0,
    T, dt,
    seed=None
):
    """
    Simulate the coupled SDE system for U(t) and S(t).

    Parameters:
    - c, a, b, beta, gamma: model parameters
    - sigma1, sigma2: noise strengths
    - U0, S0: initial conditions
    - T: total simulation time
    - dt: timestep
    - seed: random seed for reproducibility

    Returns:
    - time: time points (array)
    - U: simulated U values (array)
    - S: simulated S values (array)
    """

    if seed is not None:
        np.random.seed(seed)

    N = int(T / dt)
    time = np.linspace(0, T, N)

    U = np.zeros(N)
    S = np.zeros(N)
    U[0] = U0
    S[0] = S0

    for i in range(N-1):
        t = time[i]
        Z1 = np.random.randn()
        Z2 = np.random.randn()

        drift_U = c / (1 + np.exp(b * (t - a)))  # <-- FIXED
        U[i+1] = U[i] + (drift_U - beta * U[i]) * dt + sigma1 * np.sqrt(dt) * Z1

        drift_S = beta * U[i] - gamma * S[i]
        S[i+1] = S[i] + drift_S * dt + sigma2 * np.sqrt(dt) * Z2


    return time, U, S


if __name__ == "__main__":

    a_a = 1.0
    b_a = 0.0005
    c_a = 2.0
    beta_a = 2.35
    gamma_a = 1.0
    sigma1_a = 0.05
    sigma2_a = 0.05
    U0_a = 0.8
    S0_a = 0.8

    a_b = 0.25
    b_b = 0.0005
    c_b = 0.5
    beta_b = 2.35
    gamma_b = 1.0
    sigma1_b = 0.05
    sigma2_b = 0.05
    U0_b = 0.8
    S0_b = 0.8
    
    
    
    T= 10
    dt = 0.01

    time, U_a, S_a = simulate_sde_system(
        c_a, a_a, b_a, beta_a, gamma_a,
        sigma1_a, sigma2_a,
        U0_a, S0_a,
        T, dt,
        seed=None
    )

    time, U_b, S_b = simulate_sde_system(
        c_b, a_b, b_b, beta_b, gamma_b,
        sigma1_b, sigma2_b,
        U0_b, S0_b,
        T, dt,
        seed=None
    )
    
    results = {
        'time': time,
        'U_a': U_a,
        'S_a': S_a,
        'U_b': U_b,
        'S_b': S_b
    }
    
    # Plotting results for Model A
plt.figure(figsize=(12, 6))
plt.plot(time, U_a, label='Unspliced mRNA (U) - Gene A', color='blue')
plt.plot(time, S_a, label='Spliced mRNA (S) - Gene A', color='orange')
plt.xlabel('Time')
plt.ylabel('mRNA Concentration')
plt.title('Simulation of mRNA Dynamics - Gene A')
plt.legend()
plt.grid(True)
plt.show()

# Plotting results for Model B
plt.figure(figsize=(12, 6))
plt.plot(time, U_b, label='Unspliced mRNA (U) - Gene B', color='green')
plt.plot(time, S_b, label='Spliced mRNA (S) - Gene B', color='red')
plt.xlabel('Time')
plt.ylabel('mRNA Concentration')
plt.title('Simulation of mRNA Dynamics - Gene B')
plt.legend()
plt.grid(True)
plt.show()


 

   