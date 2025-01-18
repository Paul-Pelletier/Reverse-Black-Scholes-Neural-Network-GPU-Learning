import numpy as np
import torch
import matplotlib.pyplot as plt
from NeuralNetworkExperiment import NeuralNetwork  # Import your NeuralNetwork class


# Load scaling parameters
scaling_params = np.load("scaling_params.npy", allow_pickle=True).item()
log_fk_min = scaling_params["log_fk_min"]
log_fk_max = scaling_params["log_fk_max"]
iv_min = scaling_params["iv_min"]
iv_max = scaling_params["iv_max"]

# Normalization and Denormalization functions
def normalize_log_fk(log_fk):
    return 2 * (log_fk - log_fk_min) / (log_fk_max - log_fk_min) - 1

def denormalize_log_fk(normalized_log_fk):
    return (normalized_log_fk + 1) * (log_fk_max - log_fk_min) / 2 + log_fk_min

def normalize_iv(iv):
    return 2 * (iv - iv_min) / (iv_max - iv_min) - 1

def denormalize_iv(normalized_iv):
    return (normalized_iv + 1) * (iv_max - iv_min) / 2 + iv_min

# Black-Scholes Greeks Computation
def compute_sensitivities(log_fk_values, sigma, T, option_type=1):
    """
    Computes Black-Scholes sensitivities for a given log(F/K), sigma, and T.

    Parameters:
        log_fk_values : np.ndarray
            Array of log(F/K) values to evaluate.
        sigma : float
            Fixed volatility.
        T : float
            Fixed maturity.
        option_type : int
            1 for call, -1 for put.

    Returns:
        np.ndarray : Array of sensitivities [Delta, Gamma, Vega, Theta].
    """
    from scipy.stats import norm

    d1 = (log_fk_values + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == 1 else norm.cdf(d1) - 1
    gamma = phi_d1 / (np.exp(log_fk_values) * sigma * np.sqrt(T))
    vega = np.exp(log_fk_values) * np.sqrt(T) * phi_d1
    theta = -np.exp(log_fk_values) * phi_d1 * sigma / (2 * np.sqrt(T))

    sensitivities = np.column_stack([delta, gamma, vega, theta])
    return sensitivities

# Validation Script
def validate_model():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    input_size = 6  # Input: [Delta, Gamma, Vega, Theta, T, optionType]
    output_size = 2  # Output: [log(F/K), sigma]
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load("trained_model_dynamic.pth"))
    model.to(device)  # Move model to the device
    model.eval()

    # Generate synthetic validation data
    fixed_sigma = 0.2
    fixed_T = 1.0
    log_fk_range = np.linspace(-0.05, 0.05, 100)  # Range for log(F/K)
    sensitivities = compute_sensitivities(log_fk_range, fixed_sigma, fixed_T, option_type=1)

    # Add T and option type as constant features
    T_column = np.full((sensitivities.shape[0], 1), fixed_T)
    option_type_column = np.full((sensitivities.shape[0], 1), 1)  # Call option
    inputs = np.hstack([sensitivities, T_column, option_type_column])

    # Normalize inputs
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    # Predict using the model
    with torch.no_grad():
        predictions = model(inputs_tensor).cpu().numpy()

    # Denormalize predictions
    predicted_log_fk = denormalize_log_fk(predictions[:, 0])
    predicted_sigma = denormalize_iv(predictions[:, 1])

    # Compare predictions against true values
    mse_log_fk = np.mean((predicted_log_fk - log_fk_range)**2)
    mse_sigma = np.mean((predicted_sigma - fixed_sigma)**2)

    print(f"MSE for log(F/K): {mse_log_fk:.6f}")
    print(f"MSE for sigma: {mse_sigma:.6f}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(log_fk_range, log_fk_range, 'r--', label="True log(F/K)")
    plt.plot(log_fk_range, predicted_log_fk, 'b-', label="Predicted log(F/K)")
    plt.title("Log(F/K): True vs Predicted")
    plt.xlabel("True log(F/K)")
    plt.ylabel("Predicted log(F/K)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(log_fk_range, [1] * len(log_fk_range), 'r--', label="True sigma")
    plt.plot(log_fk_range, predicted_sigma/fixed_sigma, 'b-', label="Predicted sigma")
    plt.title("Sigma: True vs Predicted")
    plt.xlabel("True log(F/K)")
    plt.ylabel("Predicted Sigma")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate_model()
