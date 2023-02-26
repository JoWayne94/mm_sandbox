"""
Description: backprop_test.py

            Check backpropagation is working

Author(s): Haitz Sáez de Ocáriz Borde
"""


import torch
from torch import nn

import mm

in_dims = 2  # equals to input features tensor number of columns
out_dims = 2  # equals to output number of columns
in_rows = 2  # input features tensor number of rows

# Initialise nn.Linear, [rows, cols] for dimensions reference
linear_layer = nn.Linear(in_dims, out_dims)
w = linear_layer.weight  # [out_dims, in_dims]
b = linear_layer.bias  # [1, out_dims]


# Define model
class Network(torch.nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.transpose(w, 0, 1),
            requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wx = mm.matMul(x, self.weights)

        # Specify multiplication technique
        return wx.naive() + b


# Define model
model = Network()
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)
# Set up training pipeline
model.train()
optimizer.zero_grad()

# Create input
x = torch.randn(in_rows, in_dims)
# Compute output
output = model(x)
# Create target
target = torch.ones_like(output)
# Define loss function
loss_func = nn.MSELoss()
# Compute loss
loss = loss_func(target, output)

# Print current network weights
pre_update = model.weights.tolist()[0]
print('Pre-update model weights: ', pre_update)
# Backpropagate
loss.backward()
# Update model parameters
optimizer.step()
# Print current network weights
post_update = model.weights.tolist()[0]
print('Post-update model weights: ', post_update)

if pre_update == post_update:
    print('Test failed')
else:
    print('Test Passed')
