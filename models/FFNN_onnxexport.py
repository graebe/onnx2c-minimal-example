import torch
import math
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import os
from time import monotonic

device = torch.device("cpu")
dtype = torch.float
n = 2000
n_neurons = 100
epochs = 1000
lr = 0.01

x = torch.linspace(-math.pi, math.pi, n, device=device, dtype=dtype)[:, None]
y = torch.sin(x)
loss = torch.nn.MSELoss(reduction="mean")


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

pred_before = model(x)
l = loss(pred_before, y)
# print("Loss before training is: {}".format(l))

for i in range(epochs):
    #print("epoch: {}".format(i + 1))
    pred = model(x)
    l = loss(pred, y)
    
    model.zero_grad()
    l.backward()

    with torch.no_grad():
        for j, param in enumerate(model.parameters()):
            param -= lr * param.grad

model.eval()
tstart_torch = monotonic() # start execution timer
pred_after = model(x)
tend_torch = monotonic() # end execution timer
l = loss(pred_after, y)
#print("Loss after training is: {}".format(l))

plt.figure()
plt.plot(x, y, label='ref')
plt.plot(x, pred_before.detach().numpy(), label='before')
plt.plot(x, pred_after.detach().numpy(), label='after')
plt.legend()
plt.grid()
plt.title("Neural Network Fit")

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

model_onnx = onnx.load(os.path.join(export_dir, export_name))

with open(os.path.join(export_dir, 'FFNN_str.onnx'), 'w') as f:
    f.write(model_onnx.__str__())

with open(os.path.join(export_dir, 'FFNN_printable.onnx'), 'w') as f:
    f.write(onnx.helper.printable_graph(model_onnx.graph))

session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name = session.get_inputs()[0].name  
#print('Input Name:', input_name)
x_np = x.detach().numpy()
tstart_onnx = monotonic() # start execution timer
output_onnx = session.run([], {input_name: x_np[[0]]})[0]
tend_onnx = monotonic() # end execution timer
output_pytorch = model(x[[0]]).detach().numpy()

# Calculate execution times
t_torch = tend_torch - tstart_torch
t_onnx = tend_onnx - tstart_onnx

print("\n----------------------------------");
print("| Executing Python Code for FFNN |");
print("----------------------------------");

print("   Pytorch: {} mus \n   ONNX Runtime: {} mus\n".format(t_torch*1000000, t_onnx*1000000))
