#!/usr/bin/env python3

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

# Parameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
ENCODING_DIM = 100
RELU_SLOPE = 0.2
BATCH_SIZE = 16

# Visualization helper
def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)), cmap='gray', vmin=0, vmax=255)
	plt.show()

# Transforms 
data_transforms = transforms.Compose([
	transforms.Grayscale(),
	transforms.Resize((256, 256)),
	transforms.RandomCrop(128),
	transforms.ToTensor(),
	transforms.Normalize([0.5], [0.5])
])

# Dataset Initialization
train_ds = ImageFolder(root='./data/texture_1/train/', transform=data_transforms)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Display batch
#for i_batch, sample_batched in enumerate(train_dl):
#	images, _ = sample_batched 
#	if (i_batch == 0):
#		imshow(torchvision.utils.make_grid(images))
#		break

# Network definition
class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),						#Conv1 out: 64x64x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),						#Conv2 out: 32x32x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),						#Conv3 out: 32x32x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),						#Conv4 out: 16x16x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),						#Conv5 out: 16x16x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 					#Conv6 out: 8x8x128
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), 					#Conv7 out: 8x8x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),						#Conv8 out: 8x8x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.Conv2d(32, ENCODING_DIM, kernel_size=8, stride=1, padding=0), 			#Conv9 out: 1x1x100
			nn.Tanh()
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(ENCODING_DIM, 32, kernel_size=8, stride=1, padding=0),	#TConv9 out: 8x8x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),				#TConv8 out: 8x8x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),			#TConv7 out: 8x8x128
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),			#TConv6 out: 16x16x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),				#TConv5 out: 16x16x64
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),				#TConv4 out: 32x32x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),				#TConv3 out: 32x32x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),				#TConv2 out: 64x64x32
			nn.LeakyReLU(RELU_SLOPE, True),
			nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),				#TConv1 out: 128x128x1
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

# Network initialization
model = Autoencoder()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

