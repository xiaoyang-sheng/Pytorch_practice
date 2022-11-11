import torch
import torchvision.models as models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# save and load model weights
model = models.vgg16(pretrained=True)  # vgg16 as example
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16()  # no specify pretrained=True, no default pram
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

torch.save(model, 'model.pth')
model = torch.load('model.pth')