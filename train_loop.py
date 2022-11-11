# hyperparameters:
# Number of Epochs
# Batch Size
# Learning Rate
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization Loop
# Once the hyperparams are set, we could train and optimize the model, each iteration is called epoch.
# Each epoch includes two main parts:
# The Train Loop
# The Validation/Test Loop

# loss function
# To minimize the loss function we could train the model.
# Common lost functions include: nn.MSELoss for regression, nn.NLLoss for classification, nn.CrossEntropyLoss for
# the combination of them
# initialize the loss_fn
import torch
from torch import nn
from cnn_build import model
from data_loader import train_dataloader, test_dataloader

loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # the total data for train
    for number, (x, y) in enumerate(dataloader):
        # number: times of iteration, each time input batch=64 tensor(64,1,28,28)
        # calculate the error
        pred = model(x)
        loss = loss_fn(pred, y)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if number % 100 == 0:
            # every 100 times print result
            loss, current = loss.item(), number * len(x) # current=number of pics iterated,len(x)=batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # test set num
    num_batches = len(dataloader)  # max iteration times
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            # ex: test_loss=torch.tensor(1.0873)
            # pred.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches  # average error
    correct /= size  # accurate rate
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# Epoch 1
# -------------------------------
# loss: 1.040251  [    0/60000]
# loss: 1.070957  [ 6400/60000]
# loss: 0.869483  [12800/60000]
# loss: 1.033000  [19200/60000]
# loss: 0.908716  [25600/60000]
# loss: 0.930925  [32000/60000]
# loss: 0.973219  [38400/60000]
# loss: 0.913604  [44800/60000]
# loss: 0.960071  [51200/60000]
# loss: 0.904625  [57600/60000]
# Test Error:
#  Accuracy: 67.1%, Avg loss: 0.911718
#
# Epoch 2
# -------------------------------
# loss: 0.952776  [    0/60000]
# loss: 1.005409  [ 6400/60000]
# loss: 0.788150  [12800/60000]
# loss: 0.969153  [19200/60000]
# loss: 0.852390  [25600/60000]
# loss: 0.862806  [32000/60000]
# loss: 0.920238  [38400/60000]
# loss: 0.863878  [44800/60000]
# loss: 0.903000  [51200/60000]
# loss: 0.858517  [57600/60000]
# Test Error:
#  Accuracy: 68.3%, Avg loss: 0.859433
#
# Done!


