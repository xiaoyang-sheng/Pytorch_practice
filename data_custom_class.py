import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


# the custom dataset must contain three functions: __init__, __len__, and __getitem__
# in this case, the images could be saved in img_dir and the labels are stored in annotations_file
class CustomImageDataset(Dataset):
    # __init__function initialize the path of image, label and transformation.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.traget_transform = target_transform

    # __len__function returns the number of samples in the dataset
    def __len__(self):
        return len(self.img_labels)

    # __getitem__function returns one sample with a given index, and transform to tensor with read_image,
    # from self.img_labels to search corresponding label, use the transform function (if applicable) and return
    # the tensor images and labels as tuple.
    def __getitem__(self, idx):
        # iloc[:,:]slice, left closed right open，iloc[idx,0]row idx col 0 element
        # os.path.join: join the path
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.traget_transform:
            label = self.traget_transform(label)
        return image, label
