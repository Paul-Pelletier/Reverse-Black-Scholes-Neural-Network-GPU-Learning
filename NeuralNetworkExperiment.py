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
    def __init__(self, input_size, output_size, hidden_size=1024):  # Increased hidden size for complexity
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


# Training Function with Dynamic Plotting and Gradient Accumulation
def train_model(model, dataloader, optimizer, loss_fn, device, epochs, accumulation_steps=1):
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
        optimizer.zero_grad()  # Ensure gradients are cleared at the start of the epoch
        for i, (X_batch, y_batch) in enumerate(dataloader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to GPU
            with torch.cuda.amp.autocast():  # Use mixed precision
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch) / accumulation_steps  # Scale loss
            loss.backward()

            # Update weights every `accumulation_steps` batches
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps

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

    # Start with a small batch size and scale up
    initial_batch_size = 2048
    max_batch_size = 16000
    batch_size = initial_batch_size
    epochs = 10

    # Adjust learning rate based on batch size scaling
    base_lr = 0.001  # Original learning rate for smaller batch sizes
    scaling_factor = batch_size / 512
    learning_rate = base_lr * scaling_factor
    print(f"Starting with batch size {batch_size} and learning rate {learning_rate:.6f}")

    batch_size = max_batch_size
    print(f"\nTraining with batch size: {batch_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]  # Number of features
    output_size = y.shape[1]  # Number of target values (Greeks)
    model = NeuralNetwork(input_size=input_size, output_size=output_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, optimizer, loss_fn, device, epochs, accumulation_steps=1)

    # Save the model after each batch size
    torch.save(model.state_dict(), f"trained_model_batchsize_{batch_size}.pth")

    # Double the batch size and scale the learning rate accordingly
    batch_size *= 2
    learning_rate = base_lr * (batch_size / 512)

    print("Model training complete and saved!")
