# Imports
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import os
from time import monotonic

# Parameters
#device = torch.device("cpu")
#dtype = torch.float
n_X = 5
n_neurons = 3

# Generate dummy input
x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
x_tensor = torch.from_numpy(x).float()

# Pytorch Model
class gemm(torch.nn.Module):
    def __init__(self):
        super(gemm, self).__init__()
        self.l1 = torch.nn.Linear(n_X, n_neurons)

    def forward(self, x):
        x = self.l1(x)
        return x


model = gemm()

# Manually set weights
model.l1.weight.data = torch.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                  [6.0, 7.0, 8.0, 9.0, 10.0],
                                                  [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=float)).float()
model.l1.bias.data = torch.from_numpy(np.array([10000.0, 20000.0, 30000.0], dtype=float)).float()

# Switch to eval mode
model.eval()

# Test inference
y_pred = model(x_tensor)
print("\nPrediction: {} \n".format(y_pred))

# Export onnx
export_dir = "exports/onnx"
export_name = "GEMM.onnx"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
torch.onnx.export(
    model,
    x_tensor,
    os.path.join(export_dir, export_name),
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    verbose=False,
)

# Evaluate Rumtime Pytorch
tstart_torch = monotonic()  # start execution timer
pred_after = model(x_tensor)
tend_torch = monotonic()  # end execution timer

# Evaluate Runtime ONNX
model_onnx = onnx.load(os.path.join(export_dir, export_name))
session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name = session.get_inputs()[0].name
x_np = x_tensor.detach().numpy()
tstart_onnx = monotonic()  # start execution timer
output_onnx = session.run([], {input_name: x_np[[0]]})[0]
tend_onnx = monotonic()  # end execution timer
output_pytorch = model(x_tensor).detach().numpy()

# Calculate execution times
t_torch = tend_torch - tstart_torch
t_onnx = tend_onnx - tstart_onnx

# Print ececution times
print("\n----------------------------------")
print("| Executing Python Code for FFNN |")
print("----------------------------------")
print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch * 1000000, t_onnx * 1000000))
