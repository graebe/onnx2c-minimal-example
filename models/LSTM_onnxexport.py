# Imports
import torch
import math
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import os
from time import monotonic
import numpy as np

# Parameters
device = torch.device("cpu")
dtype = torch.float
n = 5 
n_sequence = 10
n_batches = 1
n_features = 5
n_neurons = 3

# Training Parameters
epochs = 1000
lr = 0.01

# Generate dummy input
x = torch.from_numpy(np.ones((n_sequence, n_batches, n_features), dtype=np.float32))
h = torch.from_numpy(np.ones((1, 1, n_neurons), dtype=np.float32))
c = torch.from_numpy(np.ones((1, 1, n_neurons), dtype=np.float32))
loss = torch.nn.MSELoss(reduction="mean")

# Pytorch Model
class LSTM(torch.nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.l = torch.nn.LSTM(input_size=n_features, hidden_size=n_neurons, num_layers=1)

    def forward(self, x, h, c):
        x, [h, c] = self.l(x, [h, c])
        return x, h, c

model = LSTM()

# Switch to eval mode
model.eval()

# Export onnx
export_dir = "exports/onnx"
export_name = "LSTM.onnx"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
torch.onnx.export(model,
                  (x, h, c),
                  os.path.join(export_dir, export_name),
                  export_params=True,
                  input_names = ['input', 'h_in', 'c_in'],
                  output_names = ['output', 'h_out', 'c_out'],
                  verbose=False)

# Evaluate Rumtime Pytorch
tstart_torch = monotonic() # start execution timer
pred_after = model(x, h, c)
tend_torch = monotonic() # end execution timer

# Evaluate Runtime ONNX
model_onnx = onnx.load(os.path.join(export_dir, export_name))
session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name_x = session.get_inputs()[0].name  
input_name_h = session.get_inputs()[1].name  
input_name_c = session.get_inputs()[2].name  
x_np = x.detach().numpy()
h_np = h.detach().numpy()
c_np = c.detach().numpy()
tstart_onnx = monotonic() # start execution timer
output_onnx = session.run([], {input_name_x: x_np, input_name_h: h_np, input_name_c: c_np})[0]
tend_onnx = monotonic() # end execution timer
pred = model(x, h, c)
y = pred[0].detach().numpy()
h = pred[1].detach().numpy()
c = pred[2].detach().numpy()

# Calculate execution times
t_torch = tend_torch - tstart_torch
t_onnx = tend_onnx - tstart_onnx

# Print ececution times
print("\n----------------------------------");
print("| Executing Python Code for LSTM |");
print("----------------------------------");
print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch*1000000, t_onnx*1000000))
