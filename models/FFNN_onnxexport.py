# Imports
import torch
import math
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import os
from time import monotonic

# Parameters
device = torch.device("cpu")
dtype = torch.float
n = 2000
n_neurons = 100
epochs = 1000
lr = 0.01

# Generate dummy input
x = torch.linspace(-math.pi, math.pi, n, device=device, dtype=dtype)[:, None]
y = torch.sin(x)
loss = torch.nn.MSELoss(reduction="mean")

# Pytorch Model
class ffnn(torch.nn.Module):

    def __init__(self):
        super(ffnn, self).__init__()
        self.l1 = torch.nn.Linear(1, n_neurons)
        self.a1 = torch.nn.Tanh()
        self.l2 = torch.nn.Linear(n_neurons, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        return x

model = ffnn()

# Switch to eval mode
model.eval()

# Export onnx
export_dir = "exports/onnx"
export_name = "FFNN.onnx"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
torch.onnx.export(model,
                  x[[0]],
                  os.path.join(export_dir, export_name),
                  export_params=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  verbose=False)

# Evaluate Rumtime Pytorch
tstart_torch = monotonic() # start execution timer
pred_after = model(x)
tend_torch = monotonic() # end execution timer

# Evaluate Runtime ONNX
model_onnx = onnx.load(os.path.join(export_dir, export_name))
session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name = session.get_inputs()[0].name  
x_np = x.detach().numpy()
tstart_onnx = monotonic() # start execution timer
output_onnx = session.run([], {input_name: x_np[[0]]})[0]
tend_onnx = monotonic() # end execution timer
output_pytorch = model(x[[0]]).detach().numpy()

# Calculate execution times
t_torch = tend_torch - tstart_torch
t_onnx = tend_onnx - tstart_onnx

# Print ececution times
print("\n----------------------------------");
print("| Executing Python Code for FFNN |");
print("----------------------------------");
print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch*1000000, t_onnx*1000000))
