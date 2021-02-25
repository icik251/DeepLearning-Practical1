import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1000, shuffle=True, num_workers=2
)

test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1000, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, dropout_rate, activation_function_type):
        super(ConvNet, self).__init__()

        if activation_function_type == "relu":
            activation_function = nn.ReLU()
        elif activation_function_type == "sigmoid":
            activation_function = nn.Sigmoid()
        elif activation_function_type == "leaky_relu":
            activation_function = nn.LeakyReLU()

        self.layer1 = nn.Sequential(
            # 3 is the rbg
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(4 * 4 * 128, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


from utils import train
import torch.optim as optim
from torch.nn import BatchNorm2d

net = ConvNet(0.5, 'relu')
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

list_of_dropout_rates = [0, 0.25, 0.5]
list_of_activation_functions = ["sigmoid", "relu", "leaky_relu"]
list_of_optimizers = ["SGD", "SGD_momentum", "Adam"]

if __name__ == "__main__":
    train(
        epochs=epochs,
        model=net,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        file_name='test'
    )
