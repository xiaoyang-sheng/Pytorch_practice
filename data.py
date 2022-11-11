import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

# viusalize the train data:
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",}
figure = plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols * rows + 1):
    sample_index = torch.randint(len(train_data),size=(1,)).item() # obtain the random label
    img,label = train_data[sample_index] # get the corresponding image and labels
    figure.add_subplot(rows,cols,i) # add object to subplot
    plt.title(labels_map[label])
    plt.axis("off") # turn off the axel
    plt.imshow(img.squeeze(),cmap='gray') # gray
plt.show()
