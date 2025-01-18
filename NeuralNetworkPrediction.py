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
def denormalize_log_fk(normalized_log_fk):
    return (normalized_log_fk + 1) * (log_fk_max - log_fk_min) / 2 + log_fk_min

def denormalize_iv(normalized_iv):
    return (normalized_iv + 1) * (iv_max - iv_min) / 2 + iv_min

# Main Plotting Script
def main():
    # Load the trained model
    input_size = 6
    output_size = 2
    model = NeuralNetwork(input_size=input_size, output_size=output_size)

    # Ensure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.load_state_dict(torch.load("trained_model_dynamic.pth"))
    model.to(device)  # Move the model to the device
    model.eval()

    # Generate synthetic test data
    generator = DataGenerator(
        logMoneynessRange=[log_fk_min, log_fk_max],
        maturityRange=[0.1, 10],
        volatilityRange=[iv_min, iv_max],
        numberOfPoints=100
    )
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    X, y = generator.get_data_for_nn()

    # Convert data to tensors and move them to the same device as the model
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predict using the model
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # Move predictions back to CPU for plotting

    # Denormalize predictions and true values
    log_fk_pred = denormalize_log_fk(predictions[:, 0])
    iv_pred = denormalize_iv(predictions[:, 1])
    log_fk_true = y[:, 0]
    iv_true = y[:, 1]

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))

    # Plot for log(F/K)
    plt.subplot(1, 2, 1)
    plt.scatter(log_fk_true, log_fk_pred, alpha=0.6, label="Predicted vs Actual")
    plt.plot([log_fk_true.min(), log_fk_true.max()],
             [log_fk_true.min(), log_fk_true.max()],
             color="red", linestyle="--", linewidth=2, label="Perfect Fit")
    plt.title("Actual vs Predicted: log(F/K)")
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
    plt.title("Actual vs Predicted: IV")
    plt.xlabel("Actual IV")
    plt.ylabel("Predicted IV")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
