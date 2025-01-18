import torch
from torchviz import make_dot
from NeuralNetworkExperiment import NeuralNetwork

# Define model
input_size = 6
output_size = 2
model = NeuralNetwork(input_size=input_size, output_size=output_size)

# Load model parameters
model.load_state_dict(torch.load("trained_model_dynamic.pth"))

# Generate dummy input
dummy_input = torch.rand(1, input_size)

# Visualize the computational graph
make_dot(model(dummy_input), params=dict(model.named_parameters())).render("network_architecture", format="png")
print("Graph saved as 'network_architecture.png'")
