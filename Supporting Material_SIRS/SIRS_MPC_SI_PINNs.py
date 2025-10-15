import sciann as sn
import time
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import casadi as ca
import os
import tensorflow

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
epochs_data_initial = 4500  # Number of epochs for the initial data training
epochs_ode_initial = 5000   # Number of epochs for the initial ODE training
epochs_data_reuse = 900    # Number of epochs for data training after reusing weights
epochs_ode_reuse = 1000     # Number of epochs for ODE training after reusing weights

# Number of experiment runs
num_runs = 1

# Weight storage variable
prev_weights_by_run = {}
for run in range(num_runs):
    prev_weights_by_run[run] = {
        'I_dr': None,
        'u_dr': None,
        'S': None
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

def compute_mean_std_without_extremes(values):
    # Compute mean and std, excluding max/min if num_runs >= 3

    values = np.asarray(values)
    if values.ndim == 1:
        # Handle 1D arrays
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
        # Handle 2D arrays
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

# Initial condition and observation generation
t_span_initial = np.arange(0, start_control + 1)
x_initial = odeint(SIRS, x0, t_span_initial, args=(gamma, beta, xi))
np.random.seed(3407)
S_initial, I_initial, R_initial = x_initial[:, 0], x_initial[:, 1], x_initial[:, 2]
I_observation_initial = np.clip(np.random.poisson(np.clip(I_initial * kappa * N, 0, None)) / (kappa * N), 0, 1)

# Initialize arrays to store true values
tf_int = int(tf)
S_true = np.zeros(tf_int + 1)
I_true = np.zeros(tf_int + 1)
R_true = np.zeros(tf_int + 1)
I_observation = np.zeros(tf_int + 1)

S_true[:start_control + 1] = S_initial
I_true[:start_control + 1] = I_initial
R_true[:start_control + 1] = R_initial
I_observation[:start_control + 1] = I_observation_initial

# Initialize arrays to store estimates from each run
S_est_runs = np.zeros((tf_int + 1, num_runs))
I_est_runs = np.zeros((tf_int + 1, num_runs))
R_est_runs = np.zeros((tf_int + 1, num_runs))
U_est_runs = np.zeros((tf_int + 1, num_runs))
beta_est_runs = np.zeros((tf_int + 1, num_runs))
time_data_used_runs = np.zeros((tf_int + 1, num_runs))
time_ode_used_runs = np.zeros((tf_int + 1, num_runs))
loss_data_runs = np.zeros((tf_int + 1, num_runs))
loss_ode_runs = np.zeros((tf_int + 1, num_runs))

# Initialize arrays to store mean and std values
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
time_data_used_mean = np.zeros(tf_int + 1)
loss_ode_mean = np.zeros(tf_int + 1)
loss_data_mean = np.zeros(tf_int + 1)

# Initialize array to store control inputs used
u_used_array = np.zeros(tf_int + 1)

# Training statistics
training_stats = {
    'weight_reused': np.zeros((tf_int + 1, num_runs), dtype=bool),
    'training_epochs_used': np.zeros((tf_int + 1, num_runs))
}

for k in range(start_control, int(tf)):
    print(f"\nTime step {k}/{int(tf)}")
    if ((k - start_control) % Ts == 0 or k == start_control) and k <= end_control:
        last_training_time = k  # Store the last training time k
        for run in range(num_runs):
            print(f"\nTime step {k}/{int(tf)}, Run {run + 1}")
            # Reset SciANN model
            sn.reset_session()

            # Update training data
            t_data_k = np.arange(0, k + 1)  # Include time step k
            t_data_sc_k = t_data_k / tf
            I_obs_k = I_observation[:k + 1].reshape(-1, 1)
            u_train_k = u_used_array[:k + 1].reshape(-1, 1)

            # Define variables and networks
            t = sn.Variable('t')

            # Step 1: Data Regression - train I and U networks
            I_dr = sn.Functional('I_dr', t, 4 * [50], output_activation='sigmoid')
            u_dr = sn.Functional('u_dr', t, 4 * [50], output_activation='sigmoid')

            # Weight reuse logic for data regression
            weight_reused = False
            if use_weight_reuse and k > start_control and prev_weights_by_run[run]['I_dr'] is not None:
                try:
                    # Load the weights from the previous time step of this run
                    I_dr.set_weights(prev_weights_by_run[run]['I_dr'])
                    u_dr.set_weights(prev_weights_by_run[run]['u_dr'])
                    weight_reused = True
                    print(f"Run {run + 1}: Successfully loaded data regression weights from the previous time step of this run")
                except Exception as e:
                    print(f"Run {run + 1}: Failed to load weights: {e}")

            training_stats['weight_reused'][k, run] = weight_reused

            # Select training epochs depending on whether weight reuse is enabled
            epochs_data_current = epochs_data_reuse if weight_reused else epochs_data_initial
            training_stats['training_epochs_used'][k, run] = epochs_data_current

            # Loss functions
            loss_data = [sn.Data(I_dr), sn.Data(u_dr)]

            # Build SciModel
            pinn_data = sn.SciModel(
                inputs=t,
                targets=loss_data,
                loss_func=loss_err,
                optimizer=optimizer)

            time1_data = time.time()
            history_data = pinn_data.train(t_data_sc_k.reshape(-1, 1),
                                           [I_obs_k, u_train_k],
                                           epochs=epochs_data_current,
                                           batch_size=10,
                                           verbose=0)
            time2_data = time.time()

            time_data_used = time2_data - time1_data
            loss_data = history_data.history['loss'][-1]

            # Save the weights of the data regression networks for the current run
            prev_weights_by_run[run]['I_dr'] = I_dr.get_weights()
            prev_weights_by_run[run]['u_dr'] = u_dr.get_weights()

            print(f"Time step {k} data regression completed, final epoch loss: {loss_data:.5e}")
            print(f"Data regression training time: {time_data_used:.2f} seconds")
            print(f"Epochs used: {epochs_data_current}, Weight reused: {weight_reused}")

            # Step 2: Integral Operation
            # Use trained model to estimate full-time I
            I_est_full = I_dr.eval(pinn_data, t_data_sc_k.reshape(-1, 1)).flatten()

            # Derive R and S from estimated I
            R_derived_full = np.zeros(int(k) + 1)
            for i_est in range(int(k)):
                R_derived_full[i_est + 1] = R_derived_full[i_est] + (gamma + u_used_array[i_est]) * I_est_full[i_est] - xi * R_derived_full[i_est]


            S_derived_full = 1.0 - I_est_full - R_derived_full

            # Build training data
            S_derived = S_derived_full[:k + 1].reshape(-1, 1)
            R_derived = R_derived_full[:k + 1].reshape(-1, 1)

            # Step 3: Physics-Informed Training
            S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
            I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid', trainable=False)
            u = sn.Functional('u', t, 4 * [50], output_activation='sigmoid', trainable=False)

            # Initialize weights with those from data regression
            I.set_weights(I_dr.get_weights())
            u.set_weights(u_dr.get_weights())

            # Weight reuse logic for physics-informed network
            physics_weight_reused = False
            if use_weight_reuse and k > start_control and prev_weights_by_run[run]['S'] is not None:
                try:
                    # Load the previous weights
                    S.set_weights(prev_weights_by_run[run]['S'])
                    physics_weight_reused = True
                    print(f"Run {run + 1}: Successfully loaded physics weights from the previous time step of this run")
                except Exception as e:
                    print(f"Run {run + 1}: Failed to load physics weights: {e}")

            # Select training epochs depending on whether weight reuse is enabled
            epochs_ode_current = epochs_ode_reuse if physics_weight_reused else epochs_ode_initial

            Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)
            R = 1.0 - I - S
            gamma_control = u + gamma

            # Initial conditions
            L_S0 = sn.rename((S - S_true[0]) * (1 - sn.sign(t)), 'L_S0')
            L_R0 = sn.rename((R - R_true[0]) * (1 - sn.sign(t)), 'L_R0')

            # ODEs
            L_dSdt = sn.rename((sn.diff(S, t) + tf * Beta * I * S - tf * xi * R), 'L_dSdt')
            L_dIdt = sn.rename((sn.diff(I, t) - tf * Beta * I * S + tf * gamma_control * I), 'L_dIdt')
            L_dRdt = sn.rename((sn.diff(R, t) - tf * gamma_control * I + tf * xi * R), 'L_dRdt')

            # Loss function
            loss_ode = [
                sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
                sn.PDE(L_S0), sn.PDE(L_R0),
                sn.Data(S), sn.Data(R)
            ]

            # Build SciModel
            pinn_ode = sn.SciModel(t, loss_ode, loss_err, optimizer)

            t_ode = np.arange(len(t_data_k))
            loss_train_ode = ['zeros'] * 5 + [(t_ode, S_derived), (t_ode, R_derived)]

            # Generate collocation points
            t_train_ode = np.random.uniform(np.log1p(0 / tf), np.log1p(k / tf), Nc).reshape(-1, 1)
            t_train_ode = np.exp(t_train_ode) - 1.

            # Combine data points and collocation points
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

            # Save the weights of the physics-informed network for the current run
            prev_weights_by_run[run]['S'] = S.get_weights()

            print(f"Time step {k} physics-informed training completed, final epoch loss: {loss_ode:.5e}")
            print(f"Physics-informed training time: {time_ode_used:.2f} seconds")
            print(f"Epochs used: {epochs_ode_current}, Physics weight reused: {physics_weight_reused}")

            # Evaluate current time step
            t_k_sc = np.array([[k / tf]]).reshape(-1, 1)

            # Get current beta, S, I, R estimates
            beta_est_val = Beta.eval(pinn_ode, t_k_sc).flatten()[0]
            U_est_full = u.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            S_est_full = S.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
            R_est_full = R.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()

            print(
                f"[Time Step {k} | Run {run + 1}] Estimated Values: beta = {beta_est_val:.10f}, S = {S_est_full[k]:.10f}, I = {I_est_full[k]:.10f}, R = {R_est_full[k]:.10f}")
            print(
                f"[Time Step {k} | Run {run + 1}] True Values:      beta = {beta:.10f}, S = {S_true[k]:.10f}, I = {I_true[k]:.10f}, R = {R_true[k]:.10f}")
            # Store estimation results into arrays
            time_data_used_runs[k, run] = time_data_used
            time_ode_used_runs[k, run] = time_ode_used
            loss_data_runs[k, run] = loss_data
            loss_ode_runs[k, run] = loss_ode

            # Store estimated SIRS and beta before or during the last Ts time steps
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

        # After all runs, compute mean and std (excluding min/max)
        S_est_mean[:k + 1], S_est_std[:k + 1] = compute_mean_std_without_extremes(S_est_runs[:k + 1, :])
        I_est_mean[:k + 1], I_est_std[:k + 1] = compute_mean_std_without_extremes(I_est_runs[:k + 1, :])
        R_est_mean[:k + 1], R_est_std[:k + 1] = compute_mean_std_without_extremes(R_est_runs[:k + 1, :])
        U_est_mean[:k + 1], U_est_std[:k + 1] = compute_mean_std_without_extremes(U_est_runs[:k + 1, :])

        # For beta, only use current time step
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
        time_data_used_mean[k], _ = compute_mean_std_without_extremes(time_data_used_runs[k, :])
        loss_ode_mean[k], _ = compute_mean_std_without_extremes(loss_ode_runs[k, :])
        loss_data_mean[k], _ = compute_mean_std_without_extremes(loss_data_runs[k, :])

        # Print weight reuse statistics
        reuse_count = np.sum(training_stats['weight_reused'][k, :])
        avg_epochs = np.mean(training_stats['training_epochs_used'][k, :])
        print(f"[Time Step {k}] Weight reuse: {reuse_count}/{num_runs}, Avg epochs: {avg_epochs:.1f}")

        # Compute optimal control u_used using mean estimates
        S_var_est, I_var_est = S_est_mean[k], I_est_mean[k]
        beta_est_val = beta_est_mean[k]

        # Define optimization variables for MPC
        u_numbers = int(np.ceil(N_pri / Ts))
        u_k_var = ca.SX.sym('u_k', u_numbers)

        # Initialize cost function and constraints
        cost_est = 0
        constraints_est = []

        # Simulate over prediction horizon with constant control over each Ts interval
        for i in range(N_pri):
            current_u_index = i // Ts
            current_u = u_k_var[current_u_index]

            gamma_u_est = gamma + current_u

            R_var_est = 1.0 - S_var_est - I_var_est
            dSdt_est = -beta_est_val * S_var_est * I_var_est + xi * R_var_est
            dIdt_est = beta_est_val * S_var_est * I_var_est - gamma_u_est * I_var_est
            dRdt_est = gamma_u_est * I_var_est - xi * R_var_est

            S_var_est = S_var_est + dSdt_est
            I_var_est = I_var_est + dIdt_est
            cost_est += alpha1 * (S_var_est - S_target) ** 2 + alpha2 * current_u ** 2
            constraints_est.extend([I_var_est - I_max])

        # Define NLP problem
        nlp_est = {'x': u_k_var, 'f': cost_est, 'g': ca.vertcat(*constraints_est)}

        # Control bounds
        lbx = [0.0] * u_numbers
        ubx = [u_max] * u_numbers

        # Constraint bounds
        lbg = [-ca.inf] * len(constraints_est)
        ubg = [0.0] * len(constraints_est)

        opts = {'print_time': False, 'ipopt': {'print_level': 0}}
        solver_est = ca.nlpsol('solver_est', 'ipopt', nlp_est, opts)

        # Solve the MPC optimization problem
        try:
            sol_est = solver_est(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            u_opt_est = sol_est['x'].full().flatten()[0]
            u_used = u_opt_est
            print(f"Time step {k}, Optimal control u_used = {u_used:.5f}")
        except RuntimeError as e:
            print(f"Time step {k} failed to solve MPC: {e}")
            break

        # Keep control value constant for Ts time steps
        end_index = min(k + Ts, tf_int + 1, end_control_int + 1)
        u_used_array[k:end_index] = u_used
        print(f"Time step {k} updated u_used_array[{k}:{end_index}] = {u_used_array[k:end_index]}")

    # Update true SIR states with applied control
    t_span = [k, k + 1]
    x0 = [S_true[k], I_true[k], R_true[k]]
    sol = odeint(SIRS_with_control, x0, t_span, args=(gamma, beta, xi, u_used_array[k]))
    S_true[k + 1], I_true[k + 1], R_true[k + 1] = sol[-1]

    # Generate new observation (same for all runs)
    I_observation[k + 1] = np.clip(np.random.poisson(np.clip(I_true[k + 1] * (kappa * N), 0, None)) / (kappa * N), 0, 1)

if True:
    # After simulation ends, perform 10 rounds of model training to estimate SIRS and beta from last_training_time+1 to tf
    for run in range(num_runs):
        print(f"\nPost-simulation, Model Training Run {run + 1}")
        sn.reset_session()

        # Update training data
        t_data_k = np.arange(0, int(tf) + 1)  # Include time step k
        t_data_sc_k = t_data_k / tf
        I_obs_k = I_observation[:int(tf) + 1].reshape(-1, 1)
        u_train_k = u_used_array[:int(tf) + 1].reshape(-1, 1)

        # Define variable and networks
        t = sn.Variable('t')

        # Step 1: Data Regression
        I_dr = sn.Functional('I_dr', t, 4 * [50], output_activation='sigmoid')
        u_dr = sn.Functional('u_dr', t, 4 * [50], output_activation='sigmoid')

        # Initialize using the most recent weights
        weight_reused = True
        I_dr.set_weights(prev_weights_by_run[run]['I_dr'])
        u_dr.set_weights(prev_weights_by_run[run]['u_dr'])
        epochs_data_current = epochs_data_reuse

        # Define loss function
        loss_data = [sn.Data(I_dr), sn.Data(u_dr)]

        # Build SciModel
        pinn_data = sn.SciModel(
            inputs=t,
            targets=loss_data,
            loss_func=loss_err,
            optimizer=optimizer)

        time1_data = time.time()
        history_data = pinn_data.train(t_data_sc_k.reshape(-1, 1),
                                    [I_obs_k, u_train_k],
                                    epochs=epochs_data_current,
                                    batch_size=10,
                                    verbose=0)
        time2_data = time.time()
        time_data_used = time2_data - time1_data
        loss_data = history_data.history['loss'][-1]
        print(f"Time step {tf_int} data regression model training completed, final epoch loss: {loss_data:.5e}")
        print(f"Data regression training time: {time_data_used:.2f} seconds")
        print(f"Epochs used: {epochs_data_current}, Weight reused: {weight_reused}")

        # Step 2: Integral Operation
        # Use trained model to estimate I values over the entire time
        I_est_full = I_dr.eval(pinn_data, t_data_sc_k.reshape(-1, 1)).flatten()

        # Estimate R and S based on I
        R_derived_full = np.zeros(tf_int + 1)
        for i_est in range(tf_int):
            R_derived_full[i_est + 1] = R_derived_full[i_est] + (gamma + u_used_array[i_est]) * I_est_full[i_est] - xi * R_derived_full[i_est]

        S_derived_full = 1.0 - I_est_full - R_derived_full

        # Build training data
        S_derived = S_derived_full[:tf_int + 1].reshape(-1, 1)
        R_derived = R_derived_full[:tf_int + 1].reshape(-1, 1)

        # Step 3: Physics-Informed Training
        S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
        I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid', trainable=False)
        u = sn.Functional('u', t, 4 * [50], output_activation='sigmoid', trainable=False)
        # Initialize I and u with weights from data regression
        I.set_weights(I_dr.get_weights())
        u.set_weights(u_dr.get_weights())

        # Initialize using the most recent physics weights
        physics_weight_reused = True
        S.set_weights(prev_weights_by_run[run]['S'])
        epochs_ode_current = epochs_ode_reuse

        Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)
        R = 1.0 - I - S
        gamma_control = u + gamma

        # Initial conditions
        L_S0 = sn.rename((S - S_true[0]) * (1 - sn.sign(t)), 'L_S0')
        L_R0 = sn.rename((R - R_true[0]) * (1 - sn.sign(t)), 'L_R0')

        # ODEs
        L_dSdt = sn.rename((sn.diff(S, t) + tf * Beta * I * S - tf * xi * R), 'L_dSdt')
        L_dIdt = sn.rename((sn.diff(I, t) - tf * Beta * I * S + tf * gamma_control * I), 'L_dIdt')
        L_dRdt = sn.rename((sn.diff(R, t) - tf * gamma_control * I + tf * xi * R), 'L_dRdt')

        # Loss function
        loss_ode = [
            sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
            sn.PDE(L_S0), sn.PDE(L_R0),
            sn.Data(S), sn.Data(R)
        ]

        # Build SciModel
        pinn_ode = sn.SciModel(t, loss_ode, loss_err, optimizer)

        t_ode = np.arange(len(t_data_k))
        loss_train_ode = ['zeros'] * 5 + [(t_ode, S_derived), (t_ode, R_derived)]

        # Generate collocation points
        t_train_ode = np.random.uniform(np.log1p(0 / tf), np.log1p(1.0), Nc).reshape(-1, 1)
        t_train_ode = np.exp(t_train_ode) - 1.

        # Combine data and collocation points
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
        print(f"Time step {tf_int} physics-informed model training completed, final epoch loss: {loss_ode:.5e}")
        print(f"Physics-informed training time: {time_ode_used:.2f} seconds")
        print(f"Epochs used: {epochs_ode_current}, Physics weight reused: {physics_weight_reused}")

        # Evaluate at final time step
        t_k_sc = np.array([[tf_int / tf]]).reshape(-1, 1)
        beta_est_val = Beta.eval(pinn_ode, t_k_sc).flatten()[0]
        U_est_full = u.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
        S_est_full = S.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()
        R_est_full = R.eval(pinn_ode, t_data_sc_k.reshape(-1, 1)).flatten()

        print(
            f"Time step {tf_int}, Run {run + 1} Estimated Values | beta: {beta_est_val:.10f}, S: {S_est_full[tf_int]:.10f}, I: {I_est_full[tf_int]:.10f}, R: {R_est_full[tf_int]:.10f}")
        print(
            f"Time step {tf_int}, Run {run + 1} True Values      | beta: {beta:.10f}, S: {S_true[tf_int]:.10f}, I: {I_true[tf_int]:.10f}, R: {R_true[tf_int]:.10f}")

        # Store estimation results from (last_training_time+1) to tf
        S_est_runs[last_training_time + 1:tf_int + 1, run] = S_est_full[last_training_time + 1:tf_int + 1]
        I_est_runs[last_training_time + 1:tf_int + 1, run] = I_est_full[last_training_time + 1:tf_int + 1]
        R_est_runs[last_training_time + 1:tf_int + 1, run] = R_est_full[last_training_time + 1:tf_int + 1]
        U_est_runs[last_training_time:tf_int + 1, run] = U_est_full[last_training_time:tf_int + 1]
        beta_est_runs[last_training_time + 1:tf_int + 1, run] = beta_est_val

        # Scalar values, store at final time step
        time_data_used_runs[tf_int, run] = time_data_used
        time_ode_used_runs[tf_int, run] = time_ode_used
        loss_data_runs[tf_int, run] = loss_data
        loss_ode_runs[tf_int, run] = loss_ode


# After all runs, compute mean and std while excluding the max and min
S_est_mean, S_est_std = compute_mean_std_without_extremes(S_est_runs)
I_est_mean, I_est_std = compute_mean_std_without_extremes(I_est_runs)
R_est_mean, R_est_std = compute_mean_std_without_extremes(R_est_runs)
U_est_mean, U_est_std = compute_mean_std_without_extremes(U_est_runs)
beta_est_mean, beta_est_std = compute_mean_std_without_extremes(beta_est_runs)
time_ode_used_mean, _ = compute_mean_std_without_extremes(time_ode_used_runs)
time_data_used_mean, _ = compute_mean_std_without_extremes(time_data_used_runs)
loss_ode_mean, _ = compute_mean_std_without_extremes(loss_ode_runs)

print("Final values in u_used_array:")
print(u_used_array)

# Create output directory
if use_weight_reuse:
    if kappa == 1:
        output_dir = 'Plot/SIRS_MPC_SI_PINNs_kappa1_reuse'
    elif kappa == 0.001:
        output_dir = 'Plot/SIRS_MPC_SI_PINNs_kappa0.001_reuse'
else:
    if kappa == 1:
        output_dir = 'Plot/SIRS_MPC_SI_PINNs_kappa1_no_reuse'
    elif kappa == 0.001:
        output_dir = 'Plot/SIRS_MPC_SI_PINNs_kappa0.001_no_reuse'

os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# Save mean and std to DataFrame
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
    'time_data_used_mean': time_data_used_mean,
    'loss_ode_mean': loss_ode_mean,
    'loss_data_mean': loss_data_mean,
})

# Save mean/std to file
output_path = os.path.join(output_dir, 'mean_std.csv')
results_mean_df.to_csv(output_path, index=False)
print(f"Mean and standard deviation results saved to '{output_path}'.")

# Save each individual run's results
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
        'time_data_used': time_data_used_runs[:, run],
        'loss_ode_ode': loss_ode_runs[:, run],
        'loss_data_ode': loss_data_runs[:, run],
        'weight_reused': training_stats['weight_reused'][:, run].astype(int),
        'training_epochs_used': training_stats['training_epochs_used'][:, run]
    })

    filename = f'{run + 1}.csv'
    file_path = os.path.join(output_dir, filename)
    results_run_df.to_csv(file_path, index=False)
    print(f"Run {run + 1} results saved to '{file_path}'.")

print("\n=== Experiment completed ===")
print(f"Results saved in: {output_dir}")
print("All data has been saved to CSV files.")