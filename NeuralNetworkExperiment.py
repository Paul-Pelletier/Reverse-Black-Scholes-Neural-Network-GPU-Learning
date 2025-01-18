import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DataGenerator import DataGenerator  # Import your DataGenerator class


# Custom Dataset for PyTorch
class OptionDataset(Dataset):
    def __init__(self, X, y, log_fk_min, log_fk_max, iv_min, iv_max):
        self.X = torch.tensor(X, dtype=torch.float32)
        log_fk = y[:, 0]
        iv = y[:, 1]
        self.y = torch.tensor(
            np.column_stack([
                2 * (log_fk - log_fk_min) / (log_fk_max - log_fk_min) - 1,  # Normalize log(F/K)
                2 * (iv - iv_min) / (iv_max - iv_min) - 1                   # Normalize IV
            ]),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


# Training Function
def train_model(model, dataloader, optimizer, loss_fn, device, epochs):
    model.to(device)
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return loss_history


# Main Calibration Script
def main():
    # Scaling parameters
    log_fk_min, log_fk_max = -0.1, 0.1
    iv_min, iv_max = 0.1, 1
    scaling_params = {
        "log_fk_min": log_fk_min,
        "log_fk_max": log_fk_max,
        "iv_min": iv_min,
        "iv_max": iv_max
    }
    np.save("scaling_params.npy", scaling_params)  # Save scaling parameters

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    print("Generating training data...")
    generator = DataGenerator(
        logMoneynessRange=[log_fk_min, log_fk_max],
        maturityRange=[0.1, 30],
        volatilityRange=[iv_min, iv_max],
        numberOfPoints=300
    )
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    X, y = generator.get_data_for_nn()

    # Split data into train and test sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Create PyTorch datasets and dataloaders
    train_dataset = OptionDataset(X_train, y_train, log_fk_min, log_fk_max, iv_min, iv_max)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=20, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]
    output_size = 2
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    epochs = 10
    loss_history = train_model(model, train_loader, optimizer, loss_fn, device, epochs)

    # Save the model
    torch.save(model.state_dict(), "trained_model_dynamic.pth")
    print("Model and scaling parameters saved successfully!")


if __name__ == "__main__":
    main()
