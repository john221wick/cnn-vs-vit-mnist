import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 2704)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = MNIST(root='./data', train=True, download=True,
                 transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

testset = MNIST(root='./data', train=False, download=True,
                transform=transform)

testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    for i, data in enumerate(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 500 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}: loss = {running_loss / 500:.3f}, accuracy = {100 * correct / total:.2f}%')
            running_loss = 0.0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            test_loss += nn.CrossEntropyLoss()(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss /= len(testloader)
    test_accuracy = 100 * test_correct / test_total
    
    print(f'Test set: loss = {test_loss:.3f}, accuracy = {test_accuracy:.2f}%')

print('Training finished.')
