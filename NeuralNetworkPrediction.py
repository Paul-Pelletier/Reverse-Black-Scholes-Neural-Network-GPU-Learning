import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from DataGenerator import DataGenerator  # Import your DataGenerator class

# Define the Neural Network (same architecture as used during training)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Main function for loading, predicting, and evaluating
def main():
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    input_size = 4  # Adjust based on the number of input features
    output_size = 4  # Adjust based on the number of target features
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    model.load_state_dict(torch.load("trained_model_bacthsize_16000.pth"))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")

    # Generate synthetic test data
    print("Generating test data...")
    generator = DataGenerator(
        logMoneynessRange=[-1, 1],
        maturityRange=[0.1, 2],
        volatilityRange=[0.1, 0.5],
        numberOfPoints=50
    )
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    X_test, y_test = generator.get_data_for_nn()

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Make predictions
    print("Making predictions...")
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()  # Move predictions to CPU

    # Evaluate predictions
    print("Evaluating predictions...")
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.6f}")

    # Plot predictions vs. true values (for the first feature, e.g., Delta)
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:100, 0], label="True Delta")  # First output feature (Delta)
    plt.plot(predictions[:100, 0], label="Predicted Delta", linestyle="--")
    plt.title("True vs. Predicted Delta")
    plt.xlabel("Sample Index")
    plt.ylabel("Delta")
    plt.legend()
    plt.show()

    # Example for real-world data (optional)
    print("Testing with real-world data...")
    real_world_data = torch.tensor([[0.1, 0.5, 0.2, 1],  # Sample 1
                                     [-0.2, 1.0, 0.25, -1]], dtype=torch.float32).to(device)
    with torch.no_grad():
        real_world_predictions = model(real_world_data).cpu().numpy()

    print("Real-World Predictions:")
    print(real_world_predictions)

if __name__ == "__main__":
    main()
