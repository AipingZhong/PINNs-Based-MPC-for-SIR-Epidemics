import sciann as sn
import time
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import casadi as ca
import os
import tensorflow

# Define parameter combinations to run
gamma_values = [0.2, 0.25, 0.3]
r_0_values = [2.0, 2.5, 3.0]
tf_values = {
    (0.2, 2.0): 60,
    (0.2, 2.5): 55,
    (0.2, 3.0): 50,
    (0.25, 2.0): 55,
    (0.25, 2.5): 45,
    (0.25, 3.0): 45,
    (0.3, 2.0): 50,
    (0.3, 2.5): 40,
    (0.3, 3.0): 40
}

epochs_data_values = {
    (0.2, 2.0): 4500,
    (0.2, 2.5): 4500,
    (0.2, 3.0): 3000,
    (0.25, 2.0): 2500,
    (0.25, 2.5): 2500,
    (0.25, 3.0): 2500,
    (0.3, 2.0): 2500,
    (0.3, 2.5): 2500,
    (0.3, 3.0): 4500
}

# Other parameters
N = 1000000.0  # Total population
kappa = 1  # Noise regulation factor
S0, I0, R0 = 1.0 - 0.001, 0.001, 0.0  # Initial conditions
x0 = [S0, I0, R0]

# MPC parameters
N_pri = 14  # Prediction horizon length
alpha1, alpha2 = 1e3, 1.0  # Cost function weights
I_max = 0.1  # Maximum infected ratio
u_max = 0.4  # Control constraint
Ts = 5  # Sampling interval
start_control = 0  # Time to start control
num_runs = 3  # Number of runs per parameter combination 3

# Training optimizer settings
loss_err = 'mse'
optimizer = 'adam'
adaptive_NTK = {'method': 'NTK', 'freq': 100}
Nc = 5000  # Number of collocation points 5000
epochs_ode = 5000  # Epochs for physics-informed training 5000

# Create output directory
if kappa == 1:
    output_dir = 'Plot/MPC_SI_PINNs_kappa1_ideal_mpc_multi_param'
elif kappa == 0.001:
    output_dir = 'Plot/MPC_SI_PINNs_kappa0.001_ideal_mpc_multi_param'

os.makedirs(output_dir, exist_ok=True)


# Define the SIR model without control
def SIR(x, t, gamma, beta):
    S, I, R = x
    lambda_val = beta * I
    dSdt = -lambda_val * S
    dIdt = lambda_val * S - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Define the SIR model with control input
def SIR_with_control(x, t, gamma, beta, u):
    S, I, R = x
    gamma_val = gamma + u
    dSdt = -beta * I * S
    dIdt = beta * I * S - gamma_val * I
    dRdt = gamma_val * I
    return [dSdt, dIdt, dRdt]


# Compute mean and std
def compute_mean_std(values):
    values = np.asarray(values)
    if values.ndim == 1:
        mean_value = np.mean(values)
        std_value = np.std(values)
        return mean_value, std_value
    elif values.ndim == 2:
        mean_values = np.mean(values, axis=1)
        std_values = np.std(values, axis=1)
        return mean_values, std_values
    else:
        raise ValueError("Input array must be 1D or 2D.")


# Loop through all parameter combinations
for gamma in gamma_values:
    for r_0 in r_0_values:
        # Calculate tf and beta for this combination
        tf = tf_values[(gamma, r_0)]
        epochs_data = epochs_data_values[(gamma, r_0)]
        beta = gamma * r_0

        # Create specific output directory for this parameter combination
        param_dir = f"{output_dir}/gamma_{gamma}_r_0_{r_0}"
        os.makedirs(param_dir, exist_ok=True)

        print(f"\n==== Running simulation with gamma={gamma}, r_0={r_0}, tf={tf}, beta={beta} ====")

        end_control = tf - 5  # Time to end control
        end_control_int = int(end_control)
        S_target = 1 / r_0  # Target susceptible population ratio
        tf_int = int(tf)

        # Generate initial conditions and observations
        t_span_initial = np.arange(0, start_control + 1)
        x_initial = odeint(SIR, x0, t_span_initial, args=(gamma, beta))
        np.random.seed(3407)
        S_initial, I_initial, R_initial = x_initial[:, 0], x_initial[:, 1], x_initial[:, 2]
        I_observation_initial = np.clip(np.random.poisson(np.clip(I_initial * kappa * N, 0, None)) / (kappa * N), 0, 1)

        # Initialize true value arrays
        S_true = np.zeros(tf_int + 1)
        I_true = np.zeros(tf_int + 1)
        R_true = np.zeros(tf_int + 1)
        I_observation = np.zeros(tf_int + 1)
        S_true[:start_control + 1] = S_initial
        I_true[:start_control + 1] = I_initial
        R_true[:start_control + 1] = R_initial
        I_observation[:start_control + 1] = I_observation_initial

        # Initialize arrays to store each run's results
        S_est_runs = np.zeros((tf_int + 1, num_runs))
        I_est_runs = np.zeros((tf_int + 1, num_runs))
        R_est_runs = np.zeros((tf_int + 1, num_runs))
        U_est_runs = np.zeros((tf_int + 1, num_runs))
        beta_est_runs = np.zeros((tf_int + 1, num_runs))
        time_ode_used_runs = np.zeros((tf_int + 1, num_runs))
        time_data_used_runs = np.zeros((tf_int + 1, num_runs))
        loss_ode_runs = np.zeros((tf_int + 1, num_runs))
        loss_data_runs = np.zeros((tf_int + 1, num_runs))

        # Initialize array to store applied control inputs
        u_used_array = np.zeros(tf_int + 1)

        # Run ideal MPC control
        for k in range(start_control, int(tf)):
            if ((k - start_control) % Ts == 0 or k == start_control) and k <= end_control:
                # Solve MPC with true values
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
                    print(f"Time Step {k} | Optimal Control Input u_used = {u_used:.5f}")
                except RuntimeError as e:
                    print(f"Time Step {k} | MPC optimization failed: {e}")
                    u_used = 0.0

                # Apply control input for Ts steps
                end_index = min(k + Ts, tf_int + 1, end_control_int + 1)
                u_used_array[k:end_index] = u_used

            # Update true SIR model states
            t_span = [k, k + 1]
            x0_step = [S_true[k], I_true[k], R_true[k]]
            sol = odeint(SIR_with_control, x0_step, t_span, args=(gamma, beta, u_used_array[k]))
            S_true[k + 1], I_true[k + 1], R_true[k + 1] = sol[-1]
            I_observation[k + 1] = np.clip(
                np.random.poisson(np.clip(I_true[k + 1] * (kappa * N), 0, None)) / (kappa * N), 0, 1)

        # After simulation ends, run the model training 'num_runs' times
        for run in range(num_runs):
            print(f"\nPost-simulation, Run {run + 1} model training for gamma={gamma}, r_0={r_0}")
            sn.reset_session()

            # Prepare training data
            t_data_k = np.arange(0, int(tf) + 1)
            t_data_sc_k = t_data_k / tf
            I_obs_k = I_observation[:int(tf) + 1].reshape(-1, 1)
            u_train_k = u_used_array[:int(tf) + 1].reshape(-1, 1)

            # Define variables and neural networks
            t = sn.Variable('t')

            # Step 1: Data Regression - train I and U networks
            I_dr = sn.Functional('I_dr', t, 4 * [50], output_activation='sigmoid')
            u_dr = sn.Functional('u_dr', t, 4 * [50], output_activation='sigmoid')

            # Loss functions
            loss_data = [sn.Data(I_dr), sn.Data(u_dr)]

            # Build SciModel
            pinn_data = sn.SciModel(t, loss_data, loss_err, optimizer)

            time1_data = time.time()
            history_data = pinn_data.train(t_data_sc_k.reshape(-1, 1),
                                           [I_obs_k, u_train_k],
                                           epochs=epochs_data,
                                           batch_size=10,
                                           verbose=0)
            time2_data = time.time()

            time_data_used = time2_data - time1_data
            loss_data = history_data.history['loss'][-1]
            print(f"Time step {tf_int} data regression completed, final epoch loss: {loss_data:.5e}")
            print(f"Data regression training time: {time_data_used:.2f} seconds")

            # Step 2: Integral Operation
            # Use trained model to estimate full-time I
            I_est_full = I_dr.eval(pinn_data, t_data_sc_k.reshape(-1, 1)).flatten()

            # Derive R and S from estimated I
            R_derived_full = np.zeros(tf_int + 1)
            for i_est in range(tf_int):
                R_derived_full[i_est + 1] = R_derived_full[i_est] + (gamma + u_used_array[i_est]) * I_est_full[i_est]

            S_derived_full = 1.0 - I_est_full - R_derived_full

            # Build training data
            S_derived = S_derived_full[:tf_int + 1].reshape(-1, 1)
            R_derived = R_derived_full[:tf_int + 1].reshape(-1, 1)

            # Step 3: Physics-Informed Training
            S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
            I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid', trainable=False)
            u = sn.Functional('u', t, 4 * [50], output_activation='sigmoid', trainable=False)

            # Initialize weights with those from data regression
            I.set_weights(I_dr.get_weights())
            u.set_weights(u_dr.get_weights())

            Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)
            R = 1.0 - I - S
            gamma_control = u + gamma

            # Initial conditions
            L_S0 = sn.rename((S - S_true[0]) * (1 - sn.sign(t)), 'L_S0')
            L_R0 = sn.rename((R - R_true[0]) * (1 - sn.sign(t)), 'L_R0')

            # ODEs
            L_dSdt = sn.rename((sn.diff(S, t) + tf * Beta * I * S), 'L_dSdt')
            L_dIdt = sn.rename((sn.diff(I, t) - tf * Beta * I * S + tf * gamma_control * I), 'L_dIdt')
            L_dRdt = sn.rename((sn.diff(R, t) - tf * gamma_control * I), 'L_dRdt')

            # Loss function
            loss_ode = [
                sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
                sn.PDE(L_S0), sn.PDE(L_R0),
                sn.Data(S), sn.Data(R)
            ]

            # Construct SciModel
            pinn_ode = sn.SciModel(t, loss_ode, loss_err, optimizer)

            t_ode = np.arange(len(t_data_k))
            loss_train_ode = ['zeros'] * 5 + [(t_ode, S_derived), (t_ode, R_derived)]

            # Generate collocation points
            t_train_ode = np.random.uniform(np.log1p(0 / tf), np.log1p(1.0), Nc).reshape(-1, 1)
            t_train_ode = np.exp(t_train_ode) - 1.

            # Combine training and collocation points
            t_train = np.concatenate([t_data_sc_k.reshape(-1, 1), t_train_ode])

            # Train model
            log_params = {'parameters': Beta, 'freq': 1}

            time1_ode = time.time()
            history_ode = pinn_ode.train(t_train,
                                         loss_train_ode,
                                         epochs=epochs_ode,
                                         batch_size=100,
                                         log_parameters=log_params,
                                         adaptive_weights=adaptive_NTK,
                                         verbose=0,
                                         stop_loss_value=1e-13)
            time2_ode = time.time()

            time_ode_used = time2_ode - time1_ode
            loss_ode = history_ode.history['loss'][-1]
            print(f"[Time Step {tf_int}] Physics-informed model training completed. Final epoch loss: {loss_ode:.5e}")
            print(f"Physics-informed training time: {time_ode_used:.2f} seconds")

            t_k_sc = np.array([[tf_int / tf]]).reshape(-1, 1)

            # Store results for time steps 0 to tf
            beta_est_val = Beta.eval(pinn_ode, t_k_sc).flatten()[0]
            U_est_full = u.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            S_est_full = S.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            R_est_full = R.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()

            S_est_runs[0:tf_int + 1, run] = S_est_full[0:tf_int + 1]
            I_est_runs[0:tf_int + 1, run] = I_est_full[0:tf_int + 1]
            R_est_runs[0:tf_int + 1, run] = R_est_full[0:tf_int + 1]
            U_est_runs[0:tf_int + 1, run] = U_est_full[0:tf_int + 1]
            beta_est_runs[0:tf_int + 1, run] = beta_est_val

            # Store time and loss
            time_data_used_runs[tf_int, run] = time_data_used
            time_ode_used_runs[tf_int, run] = time_ode_used
            loss_ode_runs[tf_int, run] = loss_ode
            loss_data_runs[tf_int, run] = loss_data

        # Compute mean and standard deviation
        S_est_mean, S_est_std = compute_mean_std(S_est_runs)
        I_est_mean, I_est_std = compute_mean_std(I_est_runs)
        R_est_mean, R_est_std = compute_mean_std(R_est_runs)
        U_est_mean, U_est_std = compute_mean_std(U_est_runs)
        beta_est_mean, beta_est_std = compute_mean_std(beta_est_runs)
        time_ode_used_mean, _ = compute_mean_std(time_ode_used_runs)
        time_data_used_mean, _ = compute_mean_std(time_data_used_runs)
        loss_ode_mean, _ = compute_mean_std(loss_ode_runs)
        loss_data_mean, _ = compute_mean_std(loss_data_runs)

        # Save mean and std DataFrame
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
            'u_actual': u_used_array,
            'time_data_used_mean': time_data_used_mean,
            'time_ode_used_mean': time_ode_used_mean,
            'loss_ode_mean': loss_ode_mean,
            'loss_data_mean': loss_data_mean
        })

        # Save mean/std to file
        output_path = os.path.join(param_dir, f'mean_std.csv')
        results_mean_df.to_csv(output_path, index=False)
        print(f"Mean and standard deviation saved to '{output_path}'.")

        # Save each run's results
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
                'u_actual': u_used_array,
                'time_ode_used': time_ode_used_runs[:, run],
                'loss_ode': loss_ode_runs[:, run],
                'time_data_used': time_data_used_runs[:, run],
                'loss_data': loss_data_runs[:, run]
            })

            filename = f'{run + 1}.csv'
            file_path = os.path.join(param_dir, filename)
            results_run_df.to_csv(file_path, index=False)
            print(f"Run {run + 1} results saved to '{file_path}'.")

print("\nAll parameter combinations have been processed and results saved.")