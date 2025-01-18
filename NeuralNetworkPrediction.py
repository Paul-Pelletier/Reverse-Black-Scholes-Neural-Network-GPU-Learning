import torch
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkExperiment import NeuralNetwork  # Import your NeuralNetwork class
from DataGenerator import DataGenerator  # Import your DataGenerator class


# Load scaling parameters
scaling_params = np.load("scaling_params.npy", allow_pickle=True).item()
log_fk_min = scaling_params["log_fk_min"]
log_fk_max = scaling_params["log_fk_max"]
iv_min = scaling_params["iv_min"]
iv_max = scaling_params["iv_max"]

# Normalization and Denormalization functions
def normalize_log_fk(log_fk):
    return 2 * (log_fk - log_fk_min) / (log_fk_max - log_fk_min) - 1

def normalize_iv(iv):
    return 2 * (iv - iv_min) / (iv_max - iv_min) - 1

def denormalize_log_fk(normalized_log_fk):
    return (normalized_log_fk + 1) * (log_fk_max - log_fk_min) / 2 + log_fk_min

def denormalize_iv(normalized_iv):
    return (normalized_iv + 1) * (iv_max - iv_min) / 2 + iv_min


def backtest_model(model, X, y, batch_size=8192):
    """
    Backtests the model predictions against actual values in smaller batches.

    Parameters:
        model : NeuralNetwork
            The trained PyTorch model.
        X : np.ndarray
            Input features (normalized).
        y : np.ndarray
            True values (normalized).
        batch_size : int
            The batch size to use during prediction.

    Returns:
        dict : Contains true and predicted values (denormalized) for both log(F/K) and IV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    num_samples = X_tensor.shape[0]
    predictions = []
    true_values = []

    # Process in batches
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X_tensor[i:i + batch_size]
            batch_y = y_tensor[i:i + batch_size]

            # Make predictions for the current batch
            batch_predictions = model(batch_X)
            predictions.append(batch_predictions.cpu().numpy())
            true_values.append(batch_y.cpu().numpy())

    # Concatenate all batches
    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)

    # Denormalize predictions and true values
    log_fk_pred = denormalize_log_fk(predictions[:, 0])
    iv_pred = denormalize_iv(predictions[:, 1])
    log_fk_true = denormalize_log_fk(true_values[:, 0])
    iv_true = denormalize_iv(true_values[:, 1])

    return {
        "log_fk_true": log_fk_true,
        "log_fk_pred": log_fk_pred,
        "iv_true": iv_true, 
        "iv_pred": iv_pred,
    }


# Plot Predicted vs Actual
def plot_backtest_results(results, title_suffix=""):
    """
    Plots Predicted vs Actual for log(F/K) and IV.

    Parameters:
        results : dict
            Contains true and predicted values for log(F/K) and IV.
        title_suffix : str
            Suffix for plot titles to differentiate scenarios.
    """
    log_fk_true = results["log_fk_true"]
    log_fk_pred = results["log_fk_pred"]
    iv_true = results["iv_true"]
    iv_pred = results["iv_pred"]

    # Plot for log(F/K)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(log_fk_true, log_fk_pred, alpha=0.6, label="Predicted vs Actual")
    plt.plot([log_fk_true.min(), log_fk_true.max()],
             [log_fk_true.min(), log_fk_true.max()],
             color="red", linestyle="--", linewidth=2, label="Perfect Fit")
    plt.title(f"Backtest Results: log(F/K) {title_suffix}")
    plt.xlabel("Actual log(F/K)")
    plt.ylabel("Predicted log(F/K)")
    plt.legend()
    plt.grid(True)

    # Plot for IV
    plt.subplot(1, 2, 2)
    plt.scatter(iv_true, iv_pred, alpha=0.6, label="Predicted vs Actual")
    plt.plot([iv_true.min(), iv_true.max()],
             [iv_true.min(), iv_true.max()],
             color="red", linestyle="--", linewidth=2, label="Perfect Fit")
    plt.title(f"Backtest Results: IV {title_suffix}")
    plt.xlabel("Actual IV")
    plt.ylabel("Predicted IV")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main Backtesting Script
def main():
    # Load the trained model
    input_size = 6
    output_size = 2
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load("trained_model_dynamic.pth"))

    # Backtest with constant volatility
    print("Generating constant volatility backtest data...")
    generator_constant = DataGenerator(
        logMoneynessRange=[log_fk_min, log_fk_max],
        maturityRange=[0.1, 10],
        volatilityRange=[iv_min, iv_max],
        numberOfPoints=100
    )
    generator_constant.generateTargetSpace()
    generator_constant.generateInitialSpace()
    X_constant, y_constant = generator_constant.get_data_for_nn()

    # Normalize inputs and targets
    X_normalized = X_constant
    y_normalized_constant = np.column_stack([
        normalize_log_fk(y_constant[:, 0]),
        normalize_iv(y_constant[:, 1]),
    ])

    # Backtest the model
    print("Backtesting model with constant volatility...")
    results_constant = backtest_model(model, X_normalized, y_normalized_constant)

    # Plot results
    plot_backtest_results(results_constant, "(Constant Volatility)")


if __name__ == "__main__":
    main()
