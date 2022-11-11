# we use transforms to manipulate the data to appropriate the train. Every TorchVision dataset has two parameters:
# transform (correction features), target_transform(correction label). torchvision.transform supports several
# common transforms
# the feature of the FashionMNIST is PIL image, the labels are integer. For training, we need to transform features
# to normalized tensor, and transform label as a one-hot tensor. Use ToTensor and Lambda to realize.
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# ToTensor( )
# ToTensor will turn one PIL image or NumPy arrays ndarray to float tensor and zoom the pixel intensity of the images
# in the range of [0,1]


# Lambda Transforms
# Lambda transforms here defines a function that turn integer to one-hot coding tensor.
# first create a all-zero tensor with 10 (the number of labels), and use scatter_ to change label_y index pos as 1.
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

# Tensor.scatter_(dim, index, src, reduce=None) will find the element with corresponding index on the dim dimension,
# and change it to src

print(torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(3), value=1))

# output: tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
