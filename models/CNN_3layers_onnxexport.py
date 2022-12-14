# Imports
import torch
from numpy.random import rand
import onnx
import onnxruntime
import os
from time import monotonic

# Parameters
n_fft = 512
n_spec = 160
n_channels = 1
n_kernel = 10
n_samples = 1
n_hidden = n_kernel * n_fft * n_spec

# Generate dummy spectrogram
x = rand(n_samples, n_channels, n_fft, n_spec)
x_tensor = torch.from_numpy(x).float()

# Pytorch Model
class CNN_3layers(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=3, kernel_size=(3, 3), padding=1, n_outputs=1):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(n_hidden, n_outputs)
        )

    def forward(self, x):
        out = self.main(x)
        return out

model = CNN_3layers(out_channels=n_kernel)

# Switch to eval mode
model.eval()

# Export onnx
export_dir = "exports/onnx"
export_name = "CNN_3layers.onnx"
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
print("\n-----------------------------------------");
print("| Executing Python Code for CNN 3 layers |");
print("-----------------------------------------");
print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch*1000000, t_onnx*1000000))
