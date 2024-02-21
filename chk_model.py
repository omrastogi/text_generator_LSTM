import torch
from model import RNN  # Assuming your RNN model is defined in a module named 'model'

# Define the model
model = RNN(input_size=100, hidden_size=256, num_layers=2, output_size=100)

# Print the model architecture
print(model)
