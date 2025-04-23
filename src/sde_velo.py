import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === FUNCTIONS ===

def hill_function_positive(p, theta, n):
    return p**n / (theta**n + p**n)

def hill_function_negative(p, theta, n):
    return theta**n / (theta**n + p**n)

def adjust_alpha(alpha, p, theta, n, feedback_type='positive'):
    if feedback_type == 'positive':
        return alpha * hill_function_positive(p, theta, n)
    elif feedback_type == 'negative':
        return alpha * hill_function_negative(p, theta, n)
    else:
        raise ValueError("feedback_type must be 'positive' or 'negative'")

def adjust_beta(beta, p, theta, n, feedback_type='positive'):
    if feedback_type == 'positive':
        return beta * hill_function_positive(p, theta, n)
    elif feedback_type == 'negative':
        return beta * hill_function_negative(p, theta, n)
    else:
        raise ValueError("feedback_type must be 'positive' or 'negative'")

def general_protein_ode(r, p, k, delta):
    return k * r - delta * p

# === SIMULATION ===

def simulate_coupled_system(
    params_a, params_b,
    T=10, dt=0.01, seed=None
):

    if seed is not None:
        np.random.seed(seed)

    N = int(T / dt)
    time = np.linspace(0, T, N)

    # Initialize arrays
    U_a = np.zeros(N)
    S_a = np.zeros(N)
    p_a = np.zeros(N)
    U_b = np.zeros(N)
    S_b = np.zeros(N)
    p_b = np.zeros(N)

    # Unpack parameters
    (c_a, a_a, b_a, beta_a, gamma_a, sigma1_a, sigma2_a,
     U0_a, S0_a, p0_a, n_a, theta_a, k_a, delta_a) = params_a

    (c_b, a_b, b_b, beta_b, gamma_b, sigma1_b, sigma2_b,
     U0_b, S0_b, p0_b, n_b, theta_b, k_b, delta_b) = params_b

    # Initial conditions
    U_a[0], S_a[0], p_a[0] = U0_a, S0_a, p0_a
    U_b[0], S_b[0], p_b[0] = U0_b, S0_b, p0_b

    for i in range(N-1):
        t = time[i]
        Z1_a = np.random.randn()
        Z2_a = np.random.randn()
        Z1_b = np.random.randn()
        Z2_b = np.random.randn()

        # --- Gene A ---
        # Protein b promotes splicing of mRNA a (positive feedback on beta_a)
        beta_a_adj = adjust_beta(beta_a, p_b[i], theta_a, n_a, feedback_type='positive')
        alpha_a = c_a / (1 + np.exp(b_a * (t - a_a)))

        drift_U_a = alpha_a - beta_a_adj * U_a[i]
        U_a[i+1] = U_a[i] + drift_U_a * dt + sigma1_a * np.sqrt(dt) * Z1_a

        drift_S_a = beta_a_adj * U_a[i] - gamma_a * S_a[i]
        S_a[i+1] = S_a[i] + drift_S_a * dt + sigma2_a * np.sqrt(dt) * Z2_a

        r_a = S_a[i]
        dp_a_dt = general_protein_ode(r_a, p_a[i], k_a, delta_a)
        p_a[i+1] = p_a[i] + dp_a_dt * dt

        # --- Gene B ---
        # Protein a inhibits splicing of mRNA b (negative feedback on beta_b)
        beta_b_adj = adjust_beta(beta_b, p_a[i], theta_b, n_b, feedback_type='negative')
        alpha_b = c_b / (1 + np.exp(b_b * (t - a_b)))

        drift_U_b = alpha_b - beta_b_adj * U_b[i]
        U_b[i+1] = U_b[i] + drift_U_b * dt + sigma1_b * np.sqrt(dt) * Z1_b

        drift_S_b = beta_b_adj * U_b[i] - gamma_b * S_b[i]
        S_b[i+1] = S_b[i] + drift_S_b * dt + sigma2_b * np.sqrt(dt) * Z2_b

        r_b = S_b[i]
        dp_b_dt = general_protein_ode(r_b, p_b[i], k_b, delta_b)
        p_b[i+1] = p_b[i] + dp_b_dt * dt

    return time, U_a, S_a, p_a, U_b, S_b, p_b


# === MAIN EXECUTION ===

if __name__ == "__main__":

    params_a = (1.0, 0.0005, 2.0, 2.35, 1.0, 0.05, 0.05,
                0.8, 0.8, 0.8, 3.0, 0.21, 1.0, 1.0)

    params_b = (0.25, 0.0005, 0.5, 2.35, 1.0, 0.05, 0.05,
                0.8, 0.8, 0.8, 3.0, 0.21, 1.0, 1.0)

    T = 10
    dt = 0.01
    n_simulations = 50

    time = np.linspace(0, T, int(T/dt))

    U_a_all = []
    S_a_all = []
    U_b_all = []
    S_b_all = []

    for run in range(n_simulations):
        _, U_a, S_a, p_a, U_b, S_b, p_b = simulate_coupled_system(
            params_a, params_b, T=T, dt=dt, seed=run  # use different seeds
        )
        U_a_all.append(U_a)
        S_a_all.append(S_a)
        U_b_all.append(U_b)
        S_b_all.append(S_b)

    U_a_all = np.array(U_a_all)
    S_a_all = np.array(S_a_all)
    U_b_all = np.array(U_b_all)
    S_b_all = np.array(S_b_all)

    # Compute mean and std
    U_a_mean = np.mean(U_a_all, axis=0)
    U_a_std = np.std(U_a_all, axis=0)

    S_a_mean = np.mean(S_a_all, axis=0)
    S_a_std = np.std(S_a_all, axis=0)

    U_b_mean = np.mean(U_b_all, axis=0)
    U_b_std = np.std(U_b_all, axis=0)

    S_b_mean = np.mean(S_b_all, axis=0)
    S_b_std = np.std(S_b_all, axis=0)

    # === PLOTS ===

plt.figure(figsize=(14, 8))

# Plot for Unspliced RNAs
plt.subplot(2, 1, 1)
plt.plot(time, U_a_mean, label="Unspliced RNA A (mean)", color="blue")
plt.fill_between(time, U_a_mean - U_a_std, U_a_mean + U_a_std, color="blue", alpha=0.3)

plt.plot(time, U_b_mean, label="Unspliced RNA B (mean)", color="green")
plt.fill_between(time, U_b_mean - U_b_std, U_b_mean + U_b_std, color="green", alpha=0.3)

plt.title("Unspliced RNA Dynamics)", fontsize=17)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Concentration", fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

# Plot for Spliced RNAs
plt.subplot(2, 1, 2)
plt.plot(time, S_a_mean, label="Spliced RNA A (mean)", color="orange")
plt.fill_between(time, S_a_mean - S_a_std, S_a_mean + S_a_std, color="orange", alpha=0.3)

plt.plot(time, S_b_mean, label="Spliced RNA B (mean)", color="red")
plt.fill_between(time, S_b_mean - S_b_std, S_b_mean + S_b_std, color="red", alpha=0.3)

plt.title("Spliced RNA Dynamics", fontsize=17)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Concentration", fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("fig/sde_dynamics.pdf")
plt.show()


