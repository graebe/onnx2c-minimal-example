# Imports
import os
import numpy as np
from numpy.random import rand
import torch
import torchvision
import onnx
import onnxruntime
from time import monotonic

# Instantiate mobilnet
model = torchvision.models.mobilenet_v2(num_classes=1)

# Generate dummy spectrogram
n_fft = 512
n_spec = 120
n_channels = 3
n_samples = 1
x = rand(n_samples, n_channels, n_fft, n_spec)
x_tensor = torch.from_numpy(x).float()

# Test prediction
y = model(x_tensor)

# Switch to eval mode
model.eval()

# Export onnx
export_dir = "exports/onnx"
export_name = "MobilnetV2.onnx"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
torch.onnx.export(model,
                  x_tensor,
                  os.path.join(export_dir, export_name),
                  export_params=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  verbose=False)

# Evaluate Rumtime Pytorch
tstart_torch = monotonic() # start execution timer
pred_after = model(x_tensor)
tend_torch = monotonic() # end execution timer

# Evaluate Runtime ONNX
model_onnx = onnx.load(os.path.join(export_dir, export_name))
session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name = session.get_inputs()[0].name  
x_np = x_tensor.detach().numpy()
tstart_onnx = monotonic() # start execution timer
output_onnx = session.run([], {input_name: x_np[[0]]})[0]
tend_onnx = monotonic() # end execution timer
output_pytorch = model(x_tensor[[0]]).detach().numpy()

# Calculate execution times
t_torch = tend_torch - tstart_torch
t_onnx = tend_onnx - tstart_onnx

# Print ececution times
print("\n----------------------------------");
print("| Executing Python Code for FFNN |");
print("----------------------------------");
print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch*1000000, t_onnx*1000000))