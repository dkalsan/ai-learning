import torch
import numpy as np
import torchvision.transforms.functional as F

class NRandomCrop(object):

	def __init__(self, output_size, n):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
		assert isinstance(n, int)
		self.n = n

	@staticmethod
	def get_params(img, output_size, n):
		w, h = img.size
		th, tw = output_size

		if w == tw and h == th:
			return np.zeros(n), np.zeros(n), h, w

		i = np.random.randint(0, h-th, size=n)
		j = np.random.randint(0, w-tw, size=n)
		return i, j, th, tw

	def __call__(self, img):

		i, j, h, w = self.get_params(img, self.output_size, self.n)

		crops = []

		for ix in range(self.n):
			crop = F.crop(img, i[ix], j[ix], h, w)
			crops.append(crop)

		return tuple(crops)