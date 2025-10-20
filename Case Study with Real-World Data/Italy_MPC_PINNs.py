import sciann as sn
import time
import numpy as np
import pandas as pd
import os
import tensorflow
import tensorflow.keras.backend as K


# Total population of Italy
N = 59619115

# Training optimizer settings
loss_err = 'mse'
optimizer = 'adam'
adaptive_NTK = {'method': 'NTK', 'freq': 100}
Nc = 5000  # Number of collocation points

# Training parameters
epochs = 5000  # Number of epochs for training

# Number of experiment runs
num_runs = 1

# Known gamma value
gamma = 1.0 / 19.0  # Fixed gamma = 1/19


def compute_mean_std_without_extremes(values):
    """Compute mean and std by removing the max and min (if num_runs >= 3)"""
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


# Custom log-mean-squared-error loss function
def log_mse(y_true, y_pred):
    epsilon = 1e-6  # Prevent log(0)
    y_true_clipped = K.clip(y_true, epsilon, None)
    y_pred_clipped = K.clip(y_pred, epsilon, None)
    return 0.01 * K.mean(K.square(K.log(y_pred_clipped) - K.log(y_true_clipped)), axis=-1)


# ==================== Data Preprocessing ====================
print("Loading and preprocessing data...")

# Load CSV data
df = pd.read_csv('Italy_covid_data_processed.csv')

print(df.columns)

# Convert date to sequential time index
df['Time'] = range(0, len(df))

# Normalize Currently Infected (I)
df['I_normalized'] = df['Currently Infected'] / N

df['I_normalized_true'] = df['Currently Infected_true'] / N

# Normalize Recovered/Removed (R)
df['R_normalized_true'] = df['Recovered/Removed_true'] / N

# Calculate and normalize Susceptible (S)
df['S_normalized_true'] = (N - df['Currently Infected_true'] - df['Recovered/Removed_true']) / N

# Extract data
time_data = df['Time'].values
S_data = df['S_normalized_true'].values
I_data = df['I_normalized'].values
I_data_true = df['I_normalized_true'].values
R_data = df['R_normalized_true'].values

# Normalize time to [0, 1]
t_max = time_data.max()
time_data_normalized = time_data / t_max

print(f"Data loaded: {len(time_data)} time steps")
print(f"Time range: {time_data.min()} to {time_data.max()}")
print(f"S range: [{S_data.min():.6f}, {S_data.max():.6f}]")
print(f"I range: [{I_data.min():.6f}, {I_data.max():.6f}]")
print(f"R range: [{R_data.min():.6f}, {R_data.max():.6f}]")
print(f"\nUsing known gamma: {gamma:.6f}")

# ==================== Training ====================
# Initialize arrays to store each run's results
num_time_steps = len(time_data)
S_est_runs = np.zeros((num_time_steps, num_runs))
I_est_runs = np.zeros((num_time_steps, num_runs))
R_est_runs = np.zeros((num_time_steps, num_runs))
beta_est_runs = np.zeros(num_runs)
time_used_runs = np.zeros(num_runs)
loss_runs = np.zeros(num_runs)

for run in range(num_runs):
    print(f"\n{'=' * 60}")
    print(f"Run {run + 1}/{num_runs}")
    print(f"{'=' * 60}")

    sn.reset_session()  # Reset SciANN model

    # Define variables and networks
    t = sn.Variable('t')
    S = sn.Functional('S', t, 4 * [50], output_activation='sigmoid')
    I = sn.Functional('I', t, 4 * [50], output_activation='sigmoid')

    # Define parameter (only parameter to estimate)
    Beta = sn.Parameter(name='Beta', inputs=t, non_neg=True)

    # R is computed from S and I
    R = 1.0 - S - I

    # Initial conditions
    L_S0 = sn.rename(10 * (S - S_data[0]) * (1 - sn.sign(t)), 'L_S0')
    L_I0 = sn.rename(10 * (I - I_data_true[0]) * (1 - sn.sign(t)), 'L_I0')
    L_R0 = sn.rename(10 * (R - R_data[0]) * (1 - sn.sign(t)), 'L_R0')

    # SIR ODEs
    # The control input u is treated as a known scalar parameter (u = 0), hence gamma + u = gamma.
    L_dSdt = sn.rename((sn.diff(S, t) + t_max * Beta * I * S), 'L_dSdt')
    L_dIdt = sn.rename((sn.diff(I, t) - t_max * Beta * I * S + t_max * gamma * I), 'L_dIdt')
    L_dRdt = sn.rename((sn.diff(R, t) - t_max * gamma * I), 'L_dRdt')

    # Loss functions
    loss_ode = [
        sn.PDE(L_dSdt), sn.PDE(L_dIdt), sn.PDE(L_dRdt),
        sn.PDE(L_S0), sn.PDE(L_I0), sn.PDE(L_R0),
        sn.Data(I)
    ]

    # Build SciModel
    pinn_ode = sn.SciModel(inputs=t, targets=loss_ode,
                               loss_func=[loss_err, loss_err, loss_err, loss_err, loss_err, loss_err,
                                          loss_err], optimizer=optimizer)

    # Prepare training data
    t_obs = np.arange(len(time_data))
    I_obs = I_data.reshape(-1, 1)

    # Only I observation data
    loss_train_ode = ['zeros'] * 6 + [(t_obs, I_obs)]

    # Generate collocation points (uniformly distributed in [0, 1])
    t_train_collocation = np.random.uniform(0.0, 1.0, Nc).reshape(-1, 1)

    # Combine observation and collocation points
    t_train = np.concatenate([time_data_normalized.reshape(-1, 1), t_train_collocation])

    # Train the model
    log_params = {'parameters': [Beta], 'freq': 100}

    print(f"Training with {epochs} epochs...")
    time_start = time.time()

    history_ode = pinn_ode.train(
        t_train,
        loss_train_ode,
        epochs=epochs,
        batch_size=100,
        log_parameters=log_params,
        adaptive_weights=adaptive_NTK,
        verbose=0,
        stop_loss_value=1e-13
    )

    time_end = time.time()
    time_used = time_end - time_start
    loss_final = history_ode.history['loss'][-1]

    print(f"\n{'=' * 60}")
    print(f"Training completed | Final Loss: {loss_final:.5e}")
    print(f"Training Time: {time_used:.2f} seconds")

    # Evaluate the model
    t_eval = time_data_normalized.reshape(-1, 1)
    S_est = S.eval(pinn_ode, t_eval).flatten()
    I_est = I.eval(pinn_ode, t_eval).flatten()
    R_est = R.eval(pinn_ode, t_eval).flatten()

    # Evaluate Beta at midpoint
    t_mid = np.array([[0.5]])
    beta_est = Beta.eval(pinn_ode, t_mid).flatten()[0]

    print(f"\nEstimated Parameters:")
    print(f"  Beta:  {beta_est:.6f}")
    print(f"  gamma: {gamma:.6f} (fixed)")
    print(f"  R0 = Beta/gamma: {beta_est / gamma:.4f}")

    # Store results
    S_est_runs[:, run] = S_est
    I_est_runs[:, run] = I_est
    R_est_runs[:, run] = R_est
    beta_est_runs[run] = beta_est
    time_used_runs[run] = time_used
    loss_runs[run] = loss_final

# ==================== Compute Statistics ====================
print(f"\n{'=' * 60}")
print("Computing statistics across all runs...")
print(f"{'=' * 60}")

S_est_mean, S_est_std = compute_mean_std_without_extremes(S_est_runs)
I_est_mean, I_est_std = compute_mean_std_without_extremes(I_est_runs)
R_est_mean, R_est_std = compute_mean_std_without_extremes(R_est_runs)
beta_est_mean, beta_est_std = compute_mean_std_without_extremes(beta_est_runs)
time_used_mean, time_used_std = compute_mean_std_without_extremes(time_used_runs)
loss_mean, loss_std = compute_mean_std_without_extremes(loss_runs)

print(f"\nFinal Statistics (excluding max and min):")
print(f"  Beta:  {beta_est_mean:.6f} ± {beta_est_std:.6f}")
print(f"  gamma: {gamma:.6f} (fixed)")
print(f"  R0:    {beta_est_mean / gamma:.4f}")
print(f"  Training time: {time_used_mean:.2f} ± {time_used_std:.2f} seconds")
print(f"  Final loss: {loss_mean:.5e} ± {loss_std:.5e}")

# ==================== Save Results ====================
output_dir = 'Plot/Italy_COVID_MPC_PINNs_OnlyI'
os.makedirs(output_dir, exist_ok=True)

# Save mean and std DataFrame
results_mean_df = pd.DataFrame({
    'Time': time_data,
    'Date': df['date'].values,
    'S_true': S_data,
    'I_true': I_data,
    'R_true': R_data,
    'S_mean': S_est_mean,
    'I_mean': I_est_mean,
    'R_mean': R_est_mean,
    'S_std': S_est_std,
    'I_std': I_est_std,
    'R_std': R_est_std,
    'beta_mean': beta_est_mean,
    'gamma_fixed': gamma,
    'beta_std': beta_est_std,
})

output_path = os.path.join(output_dir, 'mean_std.csv')
results_mean_df.to_csv(output_path, index=False)
print(f"\nMean and standard deviation saved to '{output_path}'")

# Save each run's results
for run in range(num_runs):
    results_run_df = pd.DataFrame({
        'Time': time_data,
        'Date': df['date'].values,
        'S_true': S_data,
        'I_true': I_data,
        'R_true': R_data,
        'S_est': S_est_runs[:, run],
        'I_est': I_est_runs[:, run],
        'R_est': R_est_runs[:, run],
        'beta_est': beta_est_runs[run],
        'gamma_fixed': gamma,
        'time_used': time_used_runs[run],
        'loss': loss_runs[run]
    })

    filename = f'run_{run + 1}.csv'
    file_path = os.path.join(output_dir, filename)
    results_run_df.to_csv(file_path, index=False)
    print(f"Run {run + 1} results saved to '{file_path}'")

# Save parameter summary
param_summary = pd.DataFrame({
    'Run': range(1, num_runs + 1),
    'Beta': beta_est_runs,
    'gamma': gamma,
    'R0': beta_est_runs / gamma,
    'Training_Time': time_used_runs,
    'Final_Loss': loss_runs
})

param_path = os.path.join(output_dir, 'parameter_summary.csv')
param_summary.to_csv(param_path, index=False)
print(f"Parameter summary saved to '{param_path}'")

print("\n" + "=" * 60)
print("Experiment completed!")
print(f"Results saved in: {output_dir}")
print("=" * 60)