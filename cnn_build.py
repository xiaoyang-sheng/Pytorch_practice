import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# use pytorch to build cnn to do the FashionMNIST data-centered image classification
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} device'.format(device))


# define the network class, heritage from nn.Module to build nn, including two parts:
# __init__: define network
# forward: to propagate forward
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.flatten = nn.Flatten()  # make the continuous dimension to flatten to tensor
        self.layers = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10))

    def forward(self,x):
        x = self.flatten(x)  # the input to the network is (batch_size,input)
        values = self.layers(x)
        return values


model = network()

# torch.nn.Flatten(start_dim=1, end_dim=- 1) by default to preserve one dim
# start_dim：first dim to flatten (default = 1)
# end_dim：last dim to flatten (default = -1)


# torch.nn.Flatten example:
input = torch.randn(32,1,5,5)
m = nn.Flatten()
output = m(input)
print(output.size())
# torch.Size([32, 25])
m1 = nn.Flatten(0,2)
print(m1(input).size())
# torch.Size([160, 5])

# create a network example and move towards device, output the model
model = network().to(device)
print(model)

# network(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (layers): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )


# traverse the input data and execute the model forward, but no need to call forward.
x = torch.rand(2,28,28,device=device)
value = model(x)
print(value)
print(value.size())
pred_probab = nn.Softmax(dim=1)(value)
print(pred_probab)
y_pred = pred_probab.argmax(1)
print(f'predicted class:{y_pred}')

# tensor([[-0.0355,  0.0948, -0.1048,  0.0802,  0.0177,  0.0038, -0.0281, -0.0767,
#           0.0303, -0.1290],
#         [-0.0238,  0.1298, -0.0700,  0.0861,  0.0168, -0.0418, -0.0421, -0.0772,
#           0.0369, -0.1391]], grad_fn=<AddmmBackward0>)
# torch.Size([2, 10])
# tensor([[0.0977, 0.1113, 0.0912, 0.1097, 0.1030, 0.1016, 0.0984, 0.0938, 0.1043,
#          0.0890],
#         [0.0986, 0.1149, 0.0941, 0.1100, 0.1027, 0.0968, 0.0968, 0.0935, 0.1048,
#          0.0878]], grad_fn=<SoftmaxBackward0>)
# predicted class:tensor([1, 1])


# torch.nn.Softmax(dim=None) softmax normalization
# torch.nn.Softmax example
m = nn.Softmax(dim=1)
input = torch.randn(2,3)
print(input)
output = m(input)
print(output)

# tensor([[-0.5471,  1.3495,  1.5911],
#         [-0.0185, -0.1420, -0.0556]])
# tensor([[0.0619, 0.4126, 0.5254],
#         [0.3512, 0.3104, 0.3384]])



# the model layer structure
# decompose the layer in the model, compare the input and output
input_image = torch.rand(3,28,28)
print(input_image.size())
# torch.Size([3, 28, 28])


# nn.Flatten
# make 2-D 28*28 image to 784 pixels, dim of batch is reserved (dim=0)
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
# torch.Size([3, 784])

# nn.Linear
# linear transformation

layer1 = nn.Linear(in_features=28*28,out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size( ))
# torch.Size([3, 20])

# nn.ReLU
# Non-linear correction unit
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
print(hidden1.size())

# Before ReLU: tensor([[ 0.4574, -0.5313, -0.4628, -0.9403, -0.7630,  0.1807, -0.2847, -0.2741,
#           0.0954,  0.2327,  0.4603,  0.0227, -0.1299, -0.2346, -0.1800,  0.9115,
#          -0.0870, -0.0171, -0.0064,  0.0540],
#         [ 0.0888, -0.6782, -0.2557, -0.6717, -0.4488,  0.1024, -0.3013, -0.3186,
#          -0.1338,  0.3944,  0.0704,  0.1429,  0.0521, -0.3326, -0.3113,  0.6518,
#          -0.0978, -0.0721, -0.3396,  0.4712],
#         [ 0.1781,  0.0885, -0.4775, -0.5661, -0.0099,  0.2617, -0.2678, -0.1444,
#           0.1345,  0.3259,  0.3984,  0.2392,  0.0529, -0.0349, -0.3266,  0.7488,
#          -0.3498,  0.1157,  0.0126,  0.3502]], grad_fn=<AddmmBackward0>)
#
#
# After ReLU: tensor([[0.4574, 0.0000, 0.0000, 0.0000, 0.0000, 0.1807, 0.0000, 0.0000, 0.0954,
#          0.2327, 0.4603, 0.0227, 0.0000, 0.0000, 0.0000, 0.9115, 0.0000, 0.0000,
#          0.0000, 0.0540],
#         [0.0888, 0.0000, 0.0000, 0.0000, 0.0000, 0.1024, 0.0000, 0.0000, 0.0000,
#          0.3944, 0.0704, 0.1429, 0.0521, 0.0000, 0.0000, 0.6518, 0.0000, 0.0000,
#          0.0000, 0.4712],
#         [0.1781, 0.0885, 0.0000, 0.0000, 0.0000, 0.2617, 0.0000, 0.0000, 0.1345,
#          0.3259, 0.3984, 0.2392, 0.0529, 0.0000, 0.0000, 0.7488, 0.0000, 0.1157,
#          0.0126, 0.3502]], grad_fn=<ReluBackward0>)
# torch.Size([3, 20])

# nn.Sequential
# nn.Sequential is a sequential container, the data was transported by the order pre-defined.

seq_modules = nn.Sequential(flatten,layer1,nn.ReLU(),nn.Linear(20,10))
input_image = torch.randn(3,28,28)
values1 = seq_modules(input_image)
print(values1)
# tensor([[ 0.2472,  0.2597, -0.0157,  0.3206, -0.0073,  0.1631,  0.2956,  0.0561,
#           0.2993,  0.1807],
#         [-0.0782,  0.1838, -0.0215,  0.2395, -0.0804, -0.0021,  0.0883, -0.0698,
#           0.1463, -0.0151],
#         [-0.1162,  0.0673, -0.2301,  0.1612, -0.1472, -0.0447,  0.0671, -0.2915,
#           0.3176,  0.2391]], grad_fn=<AddmmBackward0>)


# nn.Softmax
# [-\infty, \infty] -> [0,1]
softmax = nn.Softmax(dim=1)
pred_probab1 = softmax(values1)
print(pred_probab1)

# tensor([[0.1062, 0.1075, 0.0816, 0.1143, 0.0823, 0.0976, 0.1115, 0.0877, 0.1119,
#          0.0994],
#         [0.0884, 0.1148, 0.0935, 0.1214, 0.0882, 0.0954, 0.1044, 0.0891, 0.1106,
#          0.0941],
#         [0.0872, 0.1048, 0.0778, 0.1151, 0.0845, 0.0937, 0.1048, 0.0732, 0.1346,
#          0.1244]], grad_fn=<SoftmaxBackward0>)

# the model parameter
# use parameters() and named_parameters() to get the param in each layer, including weight and bias

print(f'model structure:{model}\n')

for name,param in model.named_parameters():
    print(f'layer:{name}|size"{param.size()}|param:{param[:2]}\n')

# model structure:network(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (layers): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )

# layer:layers.0.weight|size"torch.Size([512, 784])|param:tensor([[ 0.0122, -0.0204, -0.0185,  ..., -0.0196,
# 0.0257, -0.0084],
#         [-0.0066, -0.0195, -0.0199,  ..., -0.0175, -0.0007,  0.0003]],
#        grad_fn=<SliceBackward0>)
#
# layer:layers.0.bias|size"torch.Size([512])|param:tensor([0.0086, 0.0104], grad_fn=<SliceBackward0>)
#
# layer:layers.2.weight|size"torch.Size([512, 512])|param:tensor([[-0.0306, -0.0408,  0.0062,  ...,  0.0289,
# -0.0164,  0.0099],
#         [ 0.0015,  0.0052,  0.0182,  ...,  0.0431, -0.0174,  0.0049]],
#        grad_fn=<SliceBackward0>)
#
# layer:layers.2.bias|size"torch.Size([512])|param:tensor([-0.0337,  0.0294], grad_fn=<SliceBackward0>)
#
# layer:layers.4.weight|size"torch.Size([10, 512])|param:tensor([[ 0.0413,  0.0015,  0.0388,  ...,  0.0347,
# 0.0160,  0.0221],
#         [-0.0010,  0.0031,  0.0421,  ..., -0.0226,  0.0340, -0.0220]],
#        grad_fn=<SliceBackward0>)
#
# layer:layers.4.bias|size"torch.Size([10])|param:tensor([0.0210, 0.0243], grad_fn=<SliceBackward0>)

