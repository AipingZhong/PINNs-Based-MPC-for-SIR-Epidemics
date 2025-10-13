import numpy as np
import pandas as pd
from scipy.integrate import odeint
import casadi as ca
import os

# ==================== Parameter Settings ====================
gamma = 0.25
r_max = 2.5
beta = gamma * r_max
tf = 45

N = 1000000.0
kappa = 0.001
S0, I0, R0 = 1.0 - 0.001, 0.001, 0.0
x0 = [S0, I0, R0]

# ==================== MPC Parameters ====================
N_pri = 14
alpha1, alpha2 = 1e3, 1.0
I_max = 0.1
u_max = 0.4
Ts = 5
start_control = 0
end_control = tf - 5
tf_int = int(tf)
end_control_int = int(end_control)
S_target = 1 / r_max

num_runs = 1

# ==================== Create Output Directory ====================
output_dir = 'Plot/MPC_EKF_results'
os.makedirs(output_dir, exist_ok=True)


# ==================== EKF Function Definition ====================
def ekf_sir_beta_with_control(I_observed, u_observed, gamma, dt=1.0):
    """
    EKF estimator with control input.

    Returns
    -------
    x_estimates : np.ndarray
        State estimates [S, I, beta]
    P_estimates : np.ndarray
        Covariance matrices
    """
    T = len(I_observed)
    n_states = 3

    # Initialization
    I0 = I_observed[0]
    S0 = 1.0 - I0
    beta0_guess = 0.5

    x_estimates = np.zeros((T, n_states))
    x_estimates[0] = np.array([S0, I0, beta0_guess])

    P_estimates = np.zeros((T, n_states, n_states))
    P_estimates[0] = np.diag([1e-4, 1e-4, 1.0])

    # Process and measurement noise
    Q = np.diag([1e-6, 1e-6, 1e-7])
    R = np.array([[(0.1 * np.mean(I_observed)) ** 2]])
    H = np.array([[0, 1, 0]])

    # EKF main loop
    for k in range(1, T):
        x_prev = x_estimates[k - 1]
        P_prev = P_estimates[k - 1]
        S, I, beta = x_prev

        # Current control input
        u_k = u_observed[k - 1]
        gamma_eff = gamma + u_k

        # Prediction step
        S_pred = S - dt * beta * S * I
        I_pred = I + dt * (beta * S * I - gamma_eff * I)
        beta_pred = beta
        x_pred = np.array([S_pred, I_pred, beta_pred])

        # Jacobian matrix
        F = np.array([
            [1 - dt * beta * I, -dt * beta * S, -dt * S * I],
            [dt * beta * I, 1 + dt * (beta * S - gamma_eff), dt * S * I],
            [0, 0, 1]
        ])

        P_pred = F @ P_prev @ F.T + Q

        # Update step
        innovation_cov = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(innovation_cov)

        measurement_residual = I_observed[k] - H @ x_pred
        x_est = x_pred + K @ measurement_residual
        x_est = np.maximum(x_est, 0)  # Ensure non-negative

        P_est = (np.eye(n_states) - K @ H) @ P_pred

        x_estimates[k] = x_est
        P_estimates[k] = P_est

    return x_estimates, P_estimates


# ==================== SIR Model Definitions ====================
def SIR(x, t, gamma, beta):
    """SIR model without control"""
    S, I, R = x
    dSdt = -beta * I * S
    dIdt = beta * I * S - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def SIR_with_control(x, t, gamma, beta, u):
    """SIR model with control input"""
    S, I, R = x
    gamma_val = gamma + u
    dSdt = -beta * I * S
    dIdt = beta * I * S - gamma_val * I
    dRdt = gamma_val * I
    return [dSdt, dIdt, dRdt]


# ==================== Main Simulation ====================

print(f"==== MPC Control with EKF Estimation (gamma={gamma}, beta={beta}, r_max={r_max}) ====\n")

# Generate initial observed data
t_span_initial = np.arange(0, start_control + 1)
x_initial = odeint(SIR, x0, t_span_initial, args=(gamma, beta))
np.random.seed(3407)
S_initial, I_initial, R_initial = x_initial[:, 0], x_initial[:, 1], x_initial[:, 2]
I_observation_initial = np.clip(
    np.random.poisson(np.clip(I_initial * kappa * N, 0, None)) / (kappa * N), 0, 1)

# Initialize true arrays
S_true = np.zeros(tf_int + 1)
I_true = np.zeros(tf_int + 1)
R_true = np.zeros(tf_int + 1)
I_observation = np.zeros(tf_int + 1)
S_true[:start_control + 1] = S_initial
I_true[:start_control + 1] = I_initial
R_true[:start_control + 1] = R_initial
I_observation[:start_control + 1] = I_observation_initial

# Store multi-run results
S_est_runs = np.zeros((tf_int + 1, num_runs))
I_est_runs = np.zeros((tf_int + 1, num_runs))
R_est_runs = np.zeros((tf_int + 1, num_runs))
U_est_runs = np.zeros((tf_int + 1, num_runs))
beta_est_runs = np.zeros((tf_int + 1, num_runs))

u_used_array = np.zeros(tf_int + 1)

# Run ideal MPC control
print("Starting MPC control simulation...")
for k in range(start_control, int(tf)):
    if ((k - start_control) % Ts == 0 or k == start_control) and k <= end_control:
        # Solve MPC using true states
        S_var, I_var = S_true[k], I_true[k]

        u_numbers = int(np.ceil(N_pri / Ts))
        u_k_var = ca.SX.sym('u_k', u_numbers)

        cost_est = 0
        constraints_est = []

        for i in range(N_pri):
            current_u_index = i // Ts
            current_u = u_k_var[current_u_index]

            gamma_u_est = gamma + current_u
            dSdt_est = -beta * S_var * I_var
            dIdt_est = beta * S_var * I_var - gamma_u_est * I_var
            S_var += dSdt_est
            I_var += dIdt_est
            cost_est += alpha1 * (S_var - S_target) ** 2 + alpha2 * current_u ** 2
            constraints_est.append(I_var - I_max)

        nlp_est = {'x': u_k_var, 'f': cost_est, 'g': ca.vertcat(*constraints_est)}
        lbx = [0.0] * u_numbers
        ubx = [u_max] * u_numbers
        lbg = [-ca.inf] * len(constraints_est)
        ubg = [0.0] * len(constraints_est)

        opts = {'print_time': False, 'ipopt': {'print_level': 0}}
        solver_est = ca.nlpsol('solver_est', 'ipopt', nlp_est, opts)

        try:
            sol_est = solver_est(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            u_opt_est = sol_est['x'].full().flatten()[0]
            u_used = u_opt_est
            print(f"Step {k} | Optimal control input u = {u_used:.5f}")
        except RuntimeError as e:
            print(f"Step {k} | MPC optimization failed: {e}")
            u_used = 0.0

        end_index = min(k + Ts, tf_int + 1, end_control_int + 1)
        u_used_array[k:end_index] = u_used

    # Update true states
    t_span = [k, k + 1]
    x0_step = [S_true[k], I_true[k], R_true[k]]
    sol = odeint(SIR_with_control, x0_step, t_span, args=(gamma, beta, u_used_array[k]))
    S_true[k + 1], I_true[k + 1], R_true[k + 1] = sol[-1]
    I_observation[k + 1] = np.clip(
        np.random.poisson(np.clip(I_true[k + 1] * (kappa * N), 0, None)) / (kappa * N), 0, 1)

print("\nMPC control simulation completed!\n")

# EKF estimation for multiple runs
print("Starting EKF estimation (multiple runs)...")
for run in range(num_runs):
    print(f"\nEKF estimation run {run + 1}")

    np.random.seed(3407 + run)

    # Generate noisy observation data (same as PINNs)
    I_obs_noisy = np.clip(
        np.random.poisson(np.clip(I_true * kappa * N, 0, None)) / (kappa * N), 0, 1)

    # Call EKF
    x_estimates, P_estimates = ekf_sir_beta_with_control(
        I_obs_noisy, u_used_array, gamma, dt=1.0)

    # Extract estimation results
    S_est = x_estimates[:, 0]
    I_est = x_estimates[:, 1]
    beta_est = x_estimates[:, 2]
    R_est = 1.0 - S_est - I_est

    # Store results
    S_est_runs[:, run] = S_est
    I_est_runs[:, run] = I_est
    R_est_runs[:, run] = R_est
    U_est_runs[:, run] = u_used_array
    beta_est_runs[:, run] = beta_est

    print(f"  Final beta estimate: {beta_est[-1]:.4f} (True value: {beta:.4f})")

print("\nAll EKF estimation runs completed!\n")

# Compute mean and std
S_est_mean = np.mean(S_est_runs, axis=1)
S_est_std = np.std(S_est_runs, axis=1)
I_est_mean = np.mean(I_est_runs, axis=1)
I_est_std = np.std(I_est_runs, axis=1)
R_est_mean = np.mean(R_est_runs, axis=1)
R_est_std = np.std(R_est_runs, axis=1)
U_est_mean = np.mean(U_est_runs, axis=1)
U_est_std = np.std(U_est_runs, axis=1)
beta_est_mean = np.mean(beta_est_runs, axis=1)
beta_est_std = np.std(beta_est_runs, axis=1)

# Save mean and std results
results_mean_df = pd.DataFrame({
    'Time': np.arange(tf_int + 1),
    'S_true': S_true,
    'I_true': I_true,
    'R_true': R_true,
    'I_observation': I_observation,
    'S_mean': S_est_mean,
    'I_mean': I_est_mean,
    'R_mean': R_est_mean,
    'U_mean': U_est_mean,
    'S_std': S_est_std,
    'I_std': I_est_std,
    'R_std': R_est_std,
    'U_std': U_est_std,
    'beta_true': beta,
    'beta_mean': beta_est_mean,
    'beta_std': beta_est_std,
    'u_actual': u_used_array
})

output_path = os.path.join(output_dir, 'mean_std.csv')
results_mean_df.to_csv(output_path, index=False)
print(f"Mean and standard deviation results saved to '{output_path}'")

# Save detailed results for each run
for run in range(num_runs):
    results_run_df = pd.DataFrame({
        'Time': np.arange(tf_int + 1),
        'S_true': S_true,
        'I_true': I_true,
        'R_true': R_true,
        'I_observation': I_observation,
        'S_est': S_est_runs[:, run],
        'I_est': I_est_runs[:, run],
        'R_est': R_est_runs[:, run],
        'U_est': U_est_runs[:, run],
        'beta_est': beta_est_runs[:, run],
        'beta_true': beta,
        'u_actual': u_used_array
    })

    filename = f'{run + 1}.csv'
    file_path = os.path.join(output_dir, filename)
    results_run_df.to_csv(file_path, index=False)
    print(f"Run {run + 1} results saved to '{file_path}'")

print("\nAll results have been saved successfully!")
print(f"\nβ estimation summary:")
print(f"  True value: {beta:.4f}")
print(f"  Estimated mean: {beta_est_mean[-1]:.4f}")
print(f"  Estimated std: {beta_est_std[-1]:.4f}")
