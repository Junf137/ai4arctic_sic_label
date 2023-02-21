import torch
import torch.nn.functional as F

input_tensor = torch.tensor([[1, 1, 2, 2],
                             [1, 6, 6, 2],
                             [5, 6, 6, 4],
                             [5,5,4,4]]).float().view((1,1,4,4))

# Nearest interpolation with output size of (5, 5)
output_tensor_nearest = F.interpolate(input_tensor, size=2, mode='nearest').squeeze()

# Nearest-exact interpolation with output size of (5, 5)
output_tensor_nearest_exact = F.interpolate(input_tensor, size=2, mode='nearest-exact').squeeze()

print("Input tensor:\n", input_tensor)
print("Output tensor (nearest):\n", output_tensor_nearest)
print("Output tensor (nearest-exact):\n", output_tensor_nearest_exact)

#%%