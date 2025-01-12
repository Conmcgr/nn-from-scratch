import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#load data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor(), target_transform=)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor(), target_transform=)

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

#parameters
input_size = 784
hl1_size = 128
#hl2_size = 64
output_size = 10

weights_input = torch.randn(input_size, hl1_size, requires_grad=True)
weights_hl1 = torch.randn(hl1_size, hl2_size, requires_grad=True)
#weights_hl2 = torch.randn(hl2_size, output_size, requires_grad=True)

bias_input = torch.zeros(hl1_size, requires_grad=True)
bias_hl1 = torch.zeros(hl2_size, requires_grad=True)
#bias_hl2 = torch.zeros(output_size, requires_grad=True)

def softmax(x, dim):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

def relu(x):
    mask = x < 0
    x[mask] = 0

def forward_pass(data):
    flattened_data = torch.flatten(data, start_dim=1)

    input_layer = torch.mm(flattened_data, weights_input) + bias_input
    #relu(input_layer)

    hidden_1 =  torch.mm(input_layer, weights_hl1) + bias_hl1
    #relu(hidden_1)

    hidden_2 = torch.mm(hidden_1, weights_hl2) + bias_hl2

    output = softmax(hidden_2, 1)

    return output
