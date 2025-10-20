import sciann as sn
import time
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import casadi as ca
import os
import tensorflow
import tensorflow.keras.backend as K

# Define SIRS model parameters
N = 1000000.0  # Total population
kappa = 0.001  # Noise regulation factor

S0, I0, R0 = 1.0 - 0.001, 0.001, 0.0  # Initial state values
x0 = [S0, I0, R0]
gamma = 0.12  # Recovery rate
repro = 2  # Basic reproduction number
beta = 0.24  # Transmission rate
xi = 0.02  # Loss of immunity rate (known parameter)

tf = 100.0  # Total simulation time

# MPC parameter settings
N_pri = 14  # Prediction horizon length
alpha1, alpha2 = 1e3, 1.0  # Cost function weights
I_max = 0.10  # Maximum allowable infected proportion
u_max = 0.10  # Maximum control input
Ts = 5  # Sampling period
start_control = 45  # Control start time
end_control = 79  # Control end time
end_control_int = int(end_control)
S_target = 1 / 2  # Target susceptible proportion (herd immunity threshold)

# Training optimizer settings
loss_err = 'mse'
optimizer = 'adam'
adaptive_NTK = {'method': 'NTK', 'freq': 100}
Nc = 5000  # Number of collocation points 5000

# Training strategy with weight reuse
epochs_ode_initial = 5000  # Number of epochs for the initial training 5000
epochs_ode_reuse = 1000    # Number of epochs for training after reusing weights 1000

# Number of experiment runs 10
num_runs = 1

# 权重存储变量
prev_weights_by_run = {}
for run in range(num_runs):
    prev_weights_by_run[run] = {
        'S': None,
        'I': None,
        'u': None
    }
use_weight_reuse = True  # Whether to enable weight reuse


# Define the SIRS model without control
def SIRS(x, t, gamma, beta, xi):
    S, I, R = x
    dSdt = -beta * I * S + xi * R
    dIdt = beta * I * S - gamma * I
    dRdt = gamma * I - xi * R
    return [dSdt, dIdt, dRdt]


# Define the SIRS model with control input
def SIRS_with_control(x, t, gamma, beta, xi, u):
    S, I, R = x
    gamma_u = gamma + u
    dSdt = -beta * I * S + xi * R
    dIdt = beta * I * S - gamma_u * I
    dRdt = gamma_u * I - xi * R
    return [dSdt, dIdt, dRdt]


# Custom log-mean-squared-error loss function
def log_mse(y_true, y_pred):
    epsilon = 1e-6  # Prevent log(0)
    y_true_clipped = K.clip(y_true, epsilon, None)
    y_pred_clipped = K.clip(y_pred, epsilon, None)
    return 0.01 * K.mean(K.square(K.log(y_pred_clipped) - K.log(y_true_clipped)), axis=-1)


# Compute mean and std by removing the max and min (if num_runs >= 3)
def compute_mean_std_without_extremes(values):
    values = np.asarray(values)
    if values.ndim == 1:
        num_runs = len(values)
        if num_runs >= 3:
            max_index = np.argmax(values)
            min_index = np.argmin(values)
            if max_index == min_index:
                indices_to_delete = [max_index]
            else:
                indices_to_delete = [max_index, min_index]
            values_filtered = np.delete(values, indices_to_delete)
        else:
            values_filtered = values
        mean_value = np.mean(values_filtered)
        std_value = np.std(values_filtered)
        return mean_value, std_value
    elif values.ndim == 2:
        num_steps, num_runs = values.shape
        mean_values = np.zeros(num_steps)
        std_values = np.zeros(num_steps)
        for t in range(num_steps):
            values_t = values[t, :]
            if num_runs >= 3:
                max_index = np.argmax(values_t)
                min_index = np.argmin(values_t)
                if max_index == min_index:
                    indices_to_delete = [max_index]
                else:
                    indices_to_delete = [max_index, min_index]
                values_filtered = np.delete(values_t, indices_to_delete)
            else:
                values_filtered = values_t
            mean_values[t] = np.mean(values_filtered)
            std_values[t] = np.std(values_filtered)
        return mean_values, std_values
    else:
        raise ValueError("Input array must be 1D or 2D.")


# Generate initial conditions and observations
t_span_initial = np.arange(0, start_control + 1)
x_initial = odeint(SIRS, x0, t_span_initial, args=(gamma, beta, xi))
np.random.seed(3407)
S_initial, I_initial, R_initial = x_initial[:, 0], x_initial[:, 1], x_initial[:, 2]
I_observation_initial = np.clip(np.random.poisson(np.clip(I_initial * kappa * N, 0, None)) / (kappa * N), 0, 1)

# Initialize true value arrays
tf_int = int(tf)
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
loss_ode_runs = np.zeros((tf_int + 1, num_runs))

# Initialize arrays to store mean and std
S_est_mean = np.zeros(tf_int + 1)
I_est_mean = np.zeros(tf_int + 1)
R_est_mean = np.zeros(tf_int + 1)
U_est_mean = np.zeros(tf_int + 1)
beta_est_mean = np.zeros(tf_int + 1)
S_est_std = np.zeros(tf_int + 1)
I_est_std = np.zeros(tf_int + 1)
R_est_std = np.zeros(tf_int + 1)
U_est_std = np.zeros(tf_int + 1)
beta_est_std = np.zeros(tf_int + 1)
time_ode_used_mean = np.zeros(tf_int + 1)
loss_mean = np.zeros(tf_int + 1)

# Initialize array to store applied control inputs
u_used_array = np.zeros(tf_int + 1)

# Training statistics
training_stats = {
    'weight_reused': np.zeros((tf_int + 1, num_runs), dtype=bool),
    'training_epochs_used': np.zeros((tf_int + 1, num_runs))
}

for k in range(start_control, int(tf)):
    print(f"\nTime Step {k}/{int(tf)}")
    if ((k - start_control) % Ts == 0 or k == start_control) and k <= end_control:
        last_training_time = k  # Record the last training time step
        for run in range(num_runs):
            print(f"\nTime Step {k}/{int(tf)}, Run {run + 1}")
            sn.reset_session()  # Reset SciANN model

            # Update training data
            t_data_k = np.arange(0, k + 1)
            t_data_sc_k = t_data_k / tf
            I_obs_k = I_observation[:k + 1].reshape(-1, 1)
            u_train_k = u_used_array[:k + 1].reshape(-1, 1)

            # Define variables and networks
            t = sn.Variable('t')
            S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
            I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid')
            u = sn.Functional('u', t, 4 * [50], output_activation='tanh')

            # Weight reuse logic
            weight_reused = False
            if use_weight_reuse and k > start_control and prev_weights_by_run[run]['S'] is not None:
                try:
                    # Load the weights from the previous time step of this run
                    S.set_weights(prev_weights_by_run[run]['S'])
                    I.set_weights(prev_weights_by_run[run]['I'])
                    u.set_weights(prev_weights_by_run[run]['u'])
                    weight_reused = True
                    print(f"Run {run + 1}: Successfully loaded weights from the previous time step of this run")
                except Exception as e:
                    print(f"Run {run + 1}: Failed to load weights: {e}")

            training_stats['weight_reused'][k, run] = weight_reused

            # Select training epochs depending on whether weight reuse is enabled
            epochs_ode_current = epochs_ode_reuse if weight_reused else epochs_ode_initial
            training_stats['training_epochs_used'][k, run] = epochs_ode_current

            Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)
            R = 1.0 - I - S
            gamma_control = u + gamma

            # Initial conditions
            L_S0 = sn.rename((S - S_true[0]) * (1 - sn.sign(t)), 'L_S0')
            L_I0 = sn.rename((I - I_true[0]) * (1 - sn.sign(t)), 'L_I0')
            L_R0 = sn.rename((R - R_true[0]) * (1 - sn.sign(t)), 'L_R0')

            # ODEs
            L_dSdt = sn.rename((sn.diff(S, t) + tf * Beta * I * S - tf * xi * R), 'L_dSdt')
            L_dIdt = sn.rename((sn.diff(I, t) - tf * Beta * I * S + tf * gamma_control * I), 'L_dIdt')
            L_dRdt = sn.rename((sn.diff(R, t) - tf * gamma_control * I + tf * xi * R), 'L_dRdt')

            # Loss functions
            loss_ode = [
                sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
                sn.PDE(L_S0), sn.PDE(L_I0), sn.PDE(L_R0),
                sn.Data(I), sn.Data(u)
            ]

            # Build SciModel
            pinn_ode = sn.SciModel(inputs=t, targets=loss_ode,
                                   loss_func=[loss_err, loss_err, loss_err, loss_err, loss_err, loss_err,
                                              log_mse, loss_err], optimizer=optimizer)

            t_ode = np.arange(len(t_data_k))
            loss_train_ode = ['zeros'] * 6 + [(t_ode, I_obs_k), (t_ode, u_train_k)]

            # Generate collocation points
            t_train_ode = np.random.uniform(np.log1p(0 / tf), np.log1p(k / tf), Nc).reshape(-1, 1)
            t_train_ode = np.exp(t_train_ode) - 1.

            # Combine training and collocation points
            t_train = np.concatenate([t_data_sc_k.reshape(-1, 1), t_train_ode])

            # Train the model
            log_params = {'parameters': Beta, 'freq': 1}
            time1_ode = time.time()
            history_ode = pinn_ode.train(t_train,
                                         loss_train_ode,
                                         epochs=epochs_ode_current,
                                         batch_size=100,
                                         log_parameters=log_params,
                                         adaptive_weights=adaptive_NTK,
                                         verbose=0,
                                         stop_loss_value=1e-13)
            time2_ode = time.time()

            time_ode_used = time2_ode - time1_ode
            loss_ode = history_ode.history['loss'][-1]

            # Save the weights of the current run
            prev_weights_by_run[run]['S'] = S.get_weights()
            prev_weights_by_run[run]['I'] = I.get_weights()
            prev_weights_by_run[run]['u'] = u.get_weights()

            print(f"Time Step {k} | Physics-Informed Model Trained | Final Loss: {loss_ode:.5e}")
            print(f"Training Time: {time_ode_used:.2f} seconds")
            print(f"Epochs used: {epochs_ode_current}, Weight reused: {weight_reused}")

            # Evaluate the model at current time step
            t_k_sc = np.array([[k / tf]]).reshape(-1, 1)
            beta_est_val = Beta.eval(pinn_ode, t_k_sc).flatten()[0]
            U_est_full = u.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            S_est_full = S.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            I_est_full = I.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            R_est_full = R.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()

            print(
                f"[Time Step {k} | Run {run + 1}] Estimated Values: beta = {beta_est_val:.10f}, S = {S_est_full[k]:.10f}, I = {I_est_full[k]:.10f}, R = {R_est_full[k]:.10f}")
            print(
                f"[Time Step {k} | Run {run + 1}] True Values:      beta = {beta:.10f}, S = {S_true[k]:.10f}, I = {I_true[k]:.10f}, R = {R_true[k]:.10f}")

            time_ode_used_runs[k, run] = time_ode_used
            loss_ode_runs[k, run] = loss_ode

            # Save results for estimated values
            if k == start_control:
                t_est_steps = np.arange(0, start_control + 1)
                t_est_steps_u = np.arange(0, start_control)
            else:
                t_est_steps = np.arange(max(0, k - Ts + 1), k + 1)
                t_est_steps_u = np.arange(max(0, k - Ts), k)

            S_est_runs[t_est_steps, run] = S_est_full[t_est_steps]
            I_est_runs[t_est_steps, run] = I_est_full[t_est_steps]
            R_est_runs[t_est_steps, run] = R_est_full[t_est_steps]
            U_est_runs[t_est_steps_u, run] = U_est_full[t_est_steps_u]
            beta_est_runs[t_est_steps, run] = beta_est_val

        # Compute mean and std excluding extremes
        S_est_mean[:k + 1], S_est_std[:k + 1] = compute_mean_std_without_extremes(S_est_runs[:k + 1, :])
        I_est_mean[:k + 1], I_est_std[:k + 1] = compute_mean_std_without_extremes(I_est_runs[:k + 1, :])
        R_est_mean[:k + 1], R_est_std[:k + 1] = compute_mean_std_without_extremes(R_est_runs[:k + 1, :])
        U_est_mean[:k + 1], U_est_std[:k + 1] = compute_mean_std_without_extremes(U_est_runs[:k + 1, :])

        beta_values = beta_est_runs[k, :]
        if num_runs >= 3:
            max_index = np.argmax(beta_values)
            min_index = np.argmin(beta_values)
            indices_to_delete = [max_index] if max_index == min_index else [max_index, min_index]
            beta_filtered = np.delete(beta_values, indices_to_delete)
        else:
            beta_filtered = beta_values
        beta_est_mean[k] = np.mean(beta_filtered)
        beta_est_std[k] = np.std(beta_filtered)

        print(
            f"[Time Step {k}] Estimated Means:     beta = {beta_est_mean[k]:.10f}, S = {S_est_mean[k]:.10f}, I = {I_est_mean[k]:.10f}, R = {R_est_mean[k]:.10f}")
        print(
            f"[Time Step {k}] True Values:         beta = {beta:.10f}, S = {S_true[k]:.10f}, I = {I_true[k]:.10f}, R = {R_true[k]:.10f}")
        print(
            f"[Time Step {k}] Standard Deviations: beta = {beta_est_std[k]:.10f}, S = {S_est_std[k]:.10f}, I = {I_est_std[k]:.10f}, R = {R_est_std[k]:.10f}")

        time_ode_used_mean[k], _ = compute_mean_std_without_extremes(time_ode_used_runs[k, :])
        loss_mean[k], _ = compute_mean_std_without_extremes(loss_ode_runs[k, :])

        # Print weight reuse statistics
        reuse_count = np.sum(training_stats['weight_reused'][k, :])
        avg_epochs = np.mean(training_stats['training_epochs_used'][k, :])
        print(f"[Time Step {k}] Weight reuse: {reuse_count}/{num_runs}, Avg epochs: {avg_epochs:.1f}")

        # Solve MPC with estimated values
        S_var_est, I_var_est = S_est_mean[k], I_est_mean[k]
        beta_est_val = beta_est_mean[k]

        u_numbers = int(np.ceil(N_pri / Ts))
        u_k_var = ca.SX.sym('u_k', u_numbers)

        cost_est = 0
        constraints_est = []

        # Simulate over prediction horizon with constant control over each Ts interval
        for i in range(N_pri):
            current_u_index = i // Ts
            current_u = u_k_var[current_u_index]

            gamma_u_est = gamma + current_u
            # SIRS模型的状态更新
            R_var_est = 1.0 - S_var_est - I_var_est  # 计算R
            dSdt_est = -beta_est_val * S_var_est * I_var_est + xi * R_var_est
            dIdt_est = beta_est_val * S_var_est * I_var_est - gamma_u_est * I_var_est
            dRdt_est = gamma_u_est * I_var_est - xi * R_var_est

            S_var_est = S_var_est + dSdt_est
            I_var_est = I_var_est + dIdt_est
            cost_est += alpha1 * (S_var_est - S_target) ** 2 + alpha2 * current_u ** 2
            constraints_est.extend([I_var_est - I_max])

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
            break

        # Apply control input for Ts steps
        end_index = min(k + Ts, tf_int + 1, end_control_int + 1)
        u_used_array[k:end_index] = u_used
        print(f"Time Step {k} | Updated u_used_array[{k}:{end_index}] = {u_used_array[k:end_index]}")

    # Update true SIR model states
    t_span = [k, k + 1]
    x0 = [S_true[k], I_true[k], R_true[k]]
    sol = odeint(SIRS_with_control, x0, t_span, args=(gamma, beta, xi, u_used_array[k]))
    S_true[k + 1], I_true[k + 1], R_true[k + 1] = sol[-1]
    I_observation[k + 1] = np.clip(np.random.poisson(np.clip(I_true[k + 1] * (kappa * N), 0, None)) / (kappa * N), 0, 1)

if True:
    # After simulation ends, run the model training 'num_runs' times to estimate SIR and beta from (last_training_time + 1) to tf
    for run in range(num_runs):
        print(f"\nPost-simulation, Run {run + 1} model training")
        sn.reset_session()

        # Prepare training data
        t_data_k = np.arange(0, int(tf) + 1)  # include time step k
        t_data_sc_k = t_data_k / tf
        I_obs_k = I_observation[:int(tf) + 1].reshape(-1, 1)
        u_train_k = u_used_array[:int(tf) + 1].reshape(-1, 1)

        # Define variables and neural networks
        t = sn.Variable('t')

        S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
        I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid')
        u = sn.Functional('u', t, 4 * [50], output_activation='tanh')

        # Initialize using the most recent weights
        weight_reused = True
        S.set_weights(prev_weights_by_run[run]['S'])
        I.set_weights(prev_weights_by_run[run]['I'])
        u.set_weights(prev_weights_by_run[run]['u'])
        epochs_ode_current = epochs_ode_reuse

        Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)
        R = 1.0 - I - S
        gamma_control = u + gamma

        # Initial conditions
        L_S0 = sn.rename((S - S_true[0]) * (1 - sn.sign(t)), 'L_S0')
        L_I0 = sn.rename((I - I_true[0]) * (1 - sn.sign(t)), 'L_I0')
        L_R0 = sn.rename((R - R_true[0]) * (1 - sn.sign(t)), 'L_R0')

        # ODEs
        L_dSdt = sn.rename((sn.diff(S, t) + tf * Beta * I * S - tf * xi * R), 'L_dSdt')
        L_dIdt = sn.rename((sn.diff(I, t) - tf * Beta * I * S + tf * gamma_control * I), 'L_dIdt')
        L_dRdt = sn.rename((sn.diff(R, t) - tf * gamma_control * I + tf * xi * R), 'L_dRdt')

        # Loss function
        loss_ode = [
            sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
            sn.PDE(L_S0), sn.PDE(L_I0), sn.PDE(L_R0),
            sn.Data(I), sn.Data(u)
        ]

        # Construct SciModel
        pinn_ode = sn.SciModel(inputs=t, targets=loss_ode,
                               loss_func=[loss_err, loss_err, loss_err, loss_err, loss_err, loss_err,
                                          log_mse, loss_err], optimizer=optimizer)

        t_ode = np.arange(len(t_data_k))
        loss_train_ode = ['zeros'] * 6 + [(t_ode, I_obs_k), (t_ode, u_train_k)]

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
                                     epochs=epochs_ode_current,
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
        print(f"Epochs used: {epochs_ode_current}, Weight reused: {weight_reused}")

        t_k_sc = np.array([[tf_int / tf]]).reshape(-1, 1)
        beta_est_val = Beta.eval(pinn_ode, t_k_sc).flatten()[0]
        U_est_full = u.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
        S_est_full = S.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
        I_est_full = I.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
        R_est_full = R.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()

        print(f"Time step {tf_int}, Run {run + 1} Estimated Values | beta: {beta_est_val:.10f}, S: {S_est_full[tf_int]:.10f}, I: {I_est_full[tf_int]:.10f}, R: {R_est_full[tf_int]:.10f}")
        print(f"Time step {tf_int}, Run {run + 1} True Values      | beta: {beta:.10f}, S: {S_true[tf_int]:.10f}, I: {I_true[tf_int]:.10f}, R: {R_true[tf_int]:.10f}")

        # Store results for time steps (last_training_time + 1) to tf
        S_est_runs[last_training_time + 1:tf_int + 1, run] = S_est_full[last_training_time + 1:tf_int + 1]
        I_est_runs[last_training_time + 1:tf_int + 1, run] = I_est_full[last_training_time + 1:tf_int + 1]
        R_est_runs[last_training_time + 1:tf_int + 1, run] = R_est_full[last_training_time + 1:tf_int + 1]
        U_est_runs[last_training_time:tf_int + 1, run] = U_est_full[last_training_time:tf_int + 1]
        beta_est_runs[last_training_time + 1:tf_int + 1, run] = beta_est_val

        # Store final step time and loss
        time_ode_used_runs[tf_int, run] = time_ode_used
        loss_ode_runs[tf_int, run] = loss_ode


# Compute mean and standard deviation, excluding max and min values across all runs
S_est_mean, S_est_std = compute_mean_std_without_extremes(S_est_runs)
I_est_mean, I_est_std = compute_mean_std_without_extremes(I_est_runs)
R_est_mean, R_est_std = compute_mean_std_without_extremes(R_est_runs)
U_est_mean, U_est_std = compute_mean_std_without_extremes(U_est_runs)
beta_est_mean, beta_est_std = compute_mean_std_without_extremes(beta_est_runs)
time_ode_used_mean, _ = compute_mean_std_without_extremes(time_ode_used_runs)
loss_ode_mean, _ = compute_mean_std_without_extremes(loss_ode_runs)

print("Final values in u_used_array:")
print(u_used_array)

# Create output directory
if use_weight_reuse:
    if kappa == 1:
        output_dir = 'Plot/SIRS_MPC_LS_PINNs_kappa1_reuse'
    elif kappa == 0.001:
        output_dir = 'Plot/SIRS_MPC_LS_PINNs_kappa0.001_reuse'
else:
    if kappa == 1:
        output_dir = 'Plot/SIRS_MPC_LS_PINNs_kappa1_no_reuse'
    elif kappa == 0.001:
        output_dir = 'Plot/SIRS_MPC_LS_PINNs_kappa0.001_no_reuse'

os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

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
    'time_ode_used_mean': time_ode_used_mean,
    'loss_mean': loss_mean
})

# Save mean/std to file
output_path = os.path.join(output_dir, 'mean_std.csv')
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
        'weight_reused': training_stats['weight_reused'][:, run].astype(int),
        'training_epochs_used': training_stats['training_epochs_used'][:, run]
    })

    filename = f'{run + 1}.csv'
    file_path = os.path.join(output_dir, filename)
    results_run_df.to_csv(file_path, index=False)
    print(f"Run {run + 1} results saved to '{file_path}'.")

print("\n=== Experiment completed ===")
print(f"Results saved in: {output_dir}")
print("All results have been saved to CSV files.")