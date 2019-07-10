#!/usr/bin/env python3

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),				#Conv1 out: 64x64x32
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),				#Conv2 out: 32x32x32
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),				#Conv3 out: 32x32x32
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),				#Conv4 out: 16x16x64
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),				#Conv5 out: 16x16x64
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 			#Conv6 out: 8x8x128
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), 			#Conv7 out: 8x8x64
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),				#Conv8 out: 8x8x32
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(32, 100, kernel_size=8, stride=1, padding=0), 			#Conv9 out: 1x1x100
			nn.Tanh()
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(100, 32, kernel_size=8, stride=1, padding=0),	#TConv9 out: 8x8x32
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),		#TConv8 out: 8x8x64
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),	#TConv7 out: 8x8x128
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),	#TConv6 out: 16x16x64
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),		#TConv5 out: 16x16x64
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),		#TConv4 out: 32x32x32
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),		#TConv3 out: 32x32x32
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),		#TConv2 out: 64x64x32
			nn.LeakyReLU(0.2, True),
			nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),		#TConv1 out: 128x128x1
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

model = Autoencoder()
print(model)