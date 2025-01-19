# imports 
import torch
# provides utilities for working w image data
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
#%matplotlib inline

# Loading the MNIST dataset
dataset = MNIST(root = 'data/', download=True)
# print(len(dataset))
image, label = dataset[10]
plt.imshow(image, cmap = 'gray')
# print('Label:', label)

# Need to convert from image to tensor so that PyTorch can understand
# PyTorch allows us to apply these transformations as the images are loaded
transforms = v2.Compose ([
    v2.PILToTensor()
])
mnist_dataset = MNIST(root='data/', train=True, transform=transforms)
print(mnist_dataset)

image_tensor, label = mnist_dataset[0]
print(image_tensor.shape, label)

print(image_tensor[:, 10:15, 10:15])
print(torch.max(image_tensor), torch.min(image_tensor))

#visualizing the tensor
#plt.imshow(image_tensor[0, 10:15, 10:15], cmap='gray')
#plt.show()

# Here we are splitting the dataset into 2 - validation and training. Normally there is
# a third one, the test set.
train_data, validation_data = random_split(mnist_dataset, [50000,10000])
# Print the length of train and validation datasets
print("length of Train Datasets: ", len(train_data))
print("length of Validation Datasets: ", len(validation_data))

batch_size = 128
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_load = DataLoader(validation_data, batch_size, shuffle=False)