# auto grad:
# one common algorithm to train CNN is to reverse propagation, param(model_weights) is adjusted by the gradient of the
# lost function. To calculate the gradient by the pytorch engine torch.autograd. It supports any computation of gradient
# for example the one-layer, input x, param w,n and some lost functions.

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)

# tensor(2.2890, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)


# w and b are param needed to optimize, therefore need to calculate the gradient of the lost function with respect
# to these params. Sett the reqires_grad features of tensor.

print(f'gradient function for z={z.grad_fn}\n')
print(f'gradient function for loss={loss.grad_fn}\n')

# gradient function for z=<AddBackward0 object at 0x7fb47069aa30>
# gradient function for loss=<BinaryCrossEntropyWithLogitsBackward0 object at 0x7fb47069a250>

