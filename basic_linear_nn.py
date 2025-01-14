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
hl1_size = 256
#hl2_size = 64
output_size = 10

weights_input = torch.randn(input_size, hl1_size, requires_grad=False)/10
weights_hl1 = torch.randn(hl1_size, output_size, requires_grad=False)/10

bias_input = torch.zeros(hl1_size, requires_grad=False)
bias_hl1 = torch.zeros(output_size, requires_grad=False)

def softmax(x, dim):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def training_pass(data, labels, weights_input, weights_hl1, bias_input, bias_hl1):
    flattened_data = torch.flatten(data, start_dim=1)


    input_layer = torch.mm(flattened_data, weights_input) + bias_input
    hidden_1 =  torch.mm(input_layer, weights_hl1) + bias_hl1


    predictions = softmax(hidden_1, 1)

    deriv_loss_wrt_logits = predictions - labels
    deriv_loss_wrt_weights_hl1 = torch.mm(input_layer.T, deriv_loss_wrt_logits) / data.size(0)
    deriv_loss_wrt_bias_hl1 = torch.sum(deriv_loss_wrt_logits, dim=0) / data.size(0)

    deriv_loss_wrt_input_layer_outputs = torch.mm(deriv_loss_wrt_logits, weights_hl1.T)
    deriv_loss_wrt_weights_input = torch.mm(flattened_data.T, deriv_loss_wrt_input_layer_outputs) / data.size(0)
    deriv_loss_wrt_bias_input = torch.sum(deriv_loss_wrt_input_layer_outputs, dim=0) / data.size(0)

    learning_rate = 0.001

    weights_input -= learning_rate * deriv_loss_wrt_weights_input
    bias_input -= learning_rate * deriv_loss_wrt_bias_input
    weights_hl1 -= learning_rate * deriv_loss_wrt_weights_hl1
    bias_hl1 -= learning_rate * deriv_loss_wrt_bias_hl1

    return predictions

def cross_entropy_loss(predictions, labels):
    loss = -torch.sum(labels* torch.log(predictions + 1e-8)) / predictions.size(0)
    return loss

epochs = 10

for epoch in range(epochs):
    epoch_loss = 0
    for data, labels in train_dataloader:
        predictions = training_pass(data, labels, weights_input, weights_hl1, bias_input, bias_hl1)

        loss = cross_entropy_loss(predictions, labels)
        epoch_loss += loss.item()
    print(epoch_loss)
    print(len(train_dataloader))
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_dataloader)}")