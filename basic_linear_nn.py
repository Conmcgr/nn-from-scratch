import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
#import pudb; pu.db

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

#load data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor(), target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)))
testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor(), target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)))

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=64, shuffle=True)

#parameters
input_size = 784
hl1_size = 128
output_size = 10

weights_input = torch.randn(input_size, hl1_size, requires_grad=False)/10
weights_hl1 = torch.randn(hl1_size, output_size, requires_grad=False)/10

bias_input = torch.zeros(hl1_size, requires_grad=False)
bias_hl1 = torch.zeros(output_size, requires_grad=False)

def softmax(x, dim):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

def forward_pass(data, weights_input, weights_hl1, bias_input, bias_hl1):
    flattened_data = torch.flatten(data, start_dim=1)
    input_layer = torch.mm(flattened_data, weights_input) + bias_input
    hidden_1 =  torch.mm(input_layer, weights_hl1) + bias_hl1

    predictions = softmax(hidden_1, 1)

    return predictions, input_layer

def backwards_pass(data, labels, weights_input, bias_input, input_layer, weights_hl1, bias_hl1, predictions, learning_rate):
    data = torch.flatten(data, start_dim=1)
    dz1 = predictions - labels
    dw1 = torch.mm(input_layer.T, dz1) / data.size(0)
    db1 = torch.sum(dz1, dim=0) / data.size(0)

    dz0 = torch.mm(dz1, weights_hl1.T)
    dw0 = torch.mm(data.T, dz0) / data.size(0)
    db0 = torch.sum(dz0, dim=0) / data.size(0)

    weights_input -= learning_rate * dw0
    bias_input -= learning_rate * db0
    weights_hl1 -= learning_rate * dw1
    bias_hl1 -= learning_rate * db1

def cross_entropy_loss(predictions, labels):
    loss = -torch.sum(labels* torch.log(predictions + 1e-8)) / predictions.size(0)
    return loss

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    for data, labels in train_dataloader:
        predictions, input_layer = forward_pass(data, weights_input, weights_hl1, bias_input, bias_hl1)

        loss = cross_entropy_loss(predictions, labels)
        epoch_loss += loss.item()

        learning_rate = 0.001

        backwards_pass(data, labels, weights_input, bias_input, input_layer, weights_hl1, bias_hl1, predictions, learning_rate)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataloader)}")

overall_loss = 0
overall_accuracy = 0
overall_class_accuracy = torch.zeros(output_size)

for data, labels in test_dataloader:
    predictions, _ = forward_pass(data, weights_input, weights_hl1, bias_input, bias_hl1)

    loss = cross_entropy_loss(predictions, labels).item()
    overall_loss += loss

    number_correct = torch.sum(torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1))
    accuracy = number_correct / len(predictions)
    overall_accuracy += accuracy


print(f"Test Loss: {overall_loss/len(test_dataloader)}")
print(f"Test Accuracy: {overall_accuracy/len(test_dataloader)}")