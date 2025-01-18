import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from DataGenerator import DataGenerator  # Import your DataGenerator class
from NeuralNetworkExperiment import NeuralNetwork  # Import your NeuralNetwork class


# Define the PyTorch Dataset
class OptionDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Evaluate Model
def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.
    Returns the true values and predictions for benchmarking.
    """
    model.eval()
    true_values, predictions = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            predictions.append(y_pred)
            true_values.append(y_batch.numpy())

    predictions = np.vstack(predictions)
    true_values = np.vstack(true_values)
    return true_values, predictions


# Plot Results
def plot_results(true_values, predictions, feature_names):
    """
    Plots true vs. predicted values for each target feature.
    """
    num_features = true_values.shape[1]
    plt.figure(figsize=(12, 6))

    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.plot(true_values[:100, i], label=f"True {feature_names[i]}", alpha=0.7)
        plt.plot(predictions[:100, i], label=f"Predicted {feature_names[i]}", linestyle="--")
        plt.title(f"True vs Predicted {feature_names[i]}")
        plt.xlabel("Sample Index")
        plt.ylabel(feature_names[i])
        plt.legend()

    plt.tight_layout()
    plt.show()


# Calculate Metrics
def calculate_metrics(true_values, predictions):
    """
    Calculates performance metrics for benchmarking.
    """
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")


# Main Function
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    input_size = 6  # Input features: [Delta, Gamma, Vega, Theta, T, optionType]
    output_size = 2  # Outputs: [log(F/K), sigma]
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load("trained_model_dynamic.pth"))
    model.to(device)
    print("Model loaded successfully!")

    # Generate synthetic test data
    print("Generating test data...")
    generator = DataGenerator(
        logMoneynessRange=[-0.1, 0.1],
        maturityRange=[0.1, 10],
        volatilityRange=[0.1, 0.5],
        numberOfPoints=100
    )
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    X_test, y_test = generator.get_data_for_nn()

    # Create DataLoader for the test set
    test_dataset = OptionDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate the model
    print("Evaluating the model...")
    true_values, predictions = evaluate_model(model, test_loader, device)

    # Calculate and print performance metrics
    print("\nPerformance Metrics:")
    calculate_metrics(true_values, predictions)

    # Plot results
    feature_names = ["log(F/K)", "sigma"]
    plot_results(true_values, predictions, feature_names)


if __name__ == "__main__":
    main()
