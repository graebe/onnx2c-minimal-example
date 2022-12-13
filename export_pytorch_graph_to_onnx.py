import torch
import math
import matplotlib.pyplot as plt
import onnx
import onnxruntime
import os

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
print("Loss before training is: {}".format(l))

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
pred_after = model(x)
l = loss(pred_after, y)
print("Loss after training is: {}".format(l))

plt.figure()
plt.plot(x, y, label='ref')
plt.plot(x, pred_before.detach().numpy(), label='before')
plt.plot(x, pred_after.detach().numpy(), label='after')
plt.legend()
plt.grid()
plt.title("Neural Network Fit")

export_dir = "exports/onnx"
export_name = "model.onnx"
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

with open(os.path.join(export_dir, 'model_str.onnx'), 'w') as f:
    f.write(model_onnx.__str__())

with open(os.path.join(export_dir, 'model_printable.onnx'), 'w') as f:
    f.write(onnx.helper.printable_graph(model_onnx.graph))

session = onnxruntime.InferenceSession(os.path.join(export_dir, export_name), None)
input_name = session.get_inputs()[0].name  
print('Input Name:', input_name)
x_np = x.detach().numpy()
output_onnx = session.run([], {input_name: x_np[[0]]})[0]
output_pytorch = model(x[[0]]).detach().numpy()

print("Pytorch: {} / ONNX Runtime: {} / Difference: {}".format(output_pytorch, output_onnx, output_pytorch - output_onnx))

print("Debug")