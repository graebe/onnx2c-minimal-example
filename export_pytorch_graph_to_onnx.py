import torch
import torch.nn.functional as F

class FFNN(torch.nn.Module):
    def __init__(self, n):
        self.l = self.linear(10, n)
    
    def forward(self, x):
        return F.relu(self.fc2(x))

mynet = FFNN(n=10)
print(mynet)

#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 4 * 4, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#       #x = x.view(-1, 16 * 4 * 4)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
# net = Net()