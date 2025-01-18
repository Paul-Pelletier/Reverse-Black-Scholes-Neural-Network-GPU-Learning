import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DataGenerator import DataGenerator  # Import your DataGenerator class
import matplotlib.pyplot as plt
import os

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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


# Training Function with Mixed Precision
def train_model(model, dataloader, optimizer, loss_fn, device, epochs, save_path):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # Use mixed precision
    loss_history = []

    # Resume from the last checkpoint if available
    start_epoch = 0
    checkpoint_path = os.path.join(save_path, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Save the model weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch + 1}.pth"))
            print(f"Saved model weights at epoch {epoch + 1}")

        # Save checkpoint for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)

    return loss_history

# Main Calibration Script
def main():
    # Enable cuDNN benchmark for faster training
    torch.backends.cudnn.benchmark = True

    # Scaling parameters
    log_fk_min, log_fk_max = -0.2, 0.2
    iv_min, iv_max = 0.05, 0.9
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
        maturityRange=[0.05, 30],
        volatilityRange=[iv_min, iv_max],
        numberOfPoints=100
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=4000,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )

    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]
    output_size = 2
    model = NeuralNetwork(input_size=input_size, output_size=output_size, hidden_size=1024)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Save path for weights
    save_path = "model_weights"
    os.makedirs(save_path, exist_ok=True)

    # Train the model
    print("Training the model...")
    epochs = 100
    loss_history = train_model(model, train_loader, optimizer, loss_fn, device, epochs, save_path)

    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
