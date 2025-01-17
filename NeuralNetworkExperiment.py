import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
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
    def __init__(self, input_size, output_size, hidden_size=64):
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

# Training Function with Dynamic Plotting
def train_model(model, dataloader, optimizer, loss_fn, device, epochs):
    model.to(device)  # Move model to GPU
    loss_history = []  # Store loss history for plotting

    # Initialize dynamic plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Update the plot
        ax.plot(loss_history, color="blue")
        plt.pause(0.1)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Finalize the plot
    plt.ioff()
    plt.show()

# Main Script
if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data using the DataGenerator
    generator = DataGenerator(
        logMoneynessRange=[-1, 1], 
        maturityRange=[0.1, 2], 
        volatilityRange=[0.1, 0.5], 
        numberOfPoints=100
    )
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    X, y = generator.get_data_for_nn()

    # Create PyTorch Dataset and DataLoader
    dataset = OptionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)  # Larger batch size for GPU efficiency

    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]  # Number of features
    output_size = y.shape[1]  # Number of target values (Greeks)
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, optimizer, loss_fn, device, epochs=50)

    # Save the model
    torch.save(model.state_dict(), "trained_model.pth")
    print("Model training complete and saved!")
