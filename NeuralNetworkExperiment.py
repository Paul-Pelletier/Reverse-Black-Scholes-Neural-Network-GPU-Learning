import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator  # Import your DataGenerator class


# Custom Dataset for PyTorch
class OptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

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


# Main Training Script
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    print("Generating training data...")
    generator = DataGenerator(
        logMoneynessRange=[-1, 1],
        maturityRange=[0.1, 2],
        volatilityRange=[0.1, 0.5],
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
    train_dataset = OptionDataset(X_train, y_train)
    test_dataset = OptionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=10, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]  # Number of input features (6: [Delta, Gamma, Vega, Theta, T, optionType])
    output_size = y.shape[1]  # Number of target features (2: [log(F/K), sigma])
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    epochs = 10
    loss_history = train_model(model, train_loader, optimizer, loss_fn, device, epochs)

    # Save the model
    torch.save(model.state_dict(), "trained_model_dynamic.pth")
    print("Model saved successfully!")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
