from torch.utils.data import DataLoader
from data_download import train_data, test_data
import matplotlib.pyplot as plt

# dataset will index one sample with features and labels, while training the model, we usually transfer the data
# in the min batches, in every epoch we rearrange the data to reduce the overfit and use Python multiprocessing to
# accelerate the data index. The Dataloarder is the iterator that contains the contents above.

# shuffle = TRUE then the data will be arranged in every epoch
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Below is not necessary:
# use dataloader to iterate
# every iteration will return the train_features and train_labels of each batch.
train_features,train_labels = next(iter(train_dataloader))
print(f'feature batch shape:{train_features.size()}')
print(f'label batch shape:{train_labels.size()}')
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img,cmap='gray')
plt.show()
print(f'label:{label}')



