import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import copy
import time
import os
import sys
sys.path.append('..')

from common import *
import yaml
import pickle
from PIL import Image

f = open('config.yaml', 'r')
params = yaml.load(f, yaml.Loader)


class Action_dataset(torch.utils.data.Dataset):
	def __init__(self, action_groundtruth_path=params['action_file_path'], transform=None):
		super(Action_dataset, self).__init__()
		self._action_groundtruth_path = action_groundtruth_path
		action_groundtruth = open(self._action_groundtruth_path + '/action.txt', 'r')
		self._data = []
		for line in action_groundtruth:
			line = line.rstrip()
			words = line.split()
			self._data.append((words[0], int(words[1])))

		self._transform = transform

	def unpickle(self, file):
	    with open(file, 'rb') as fo:
	        dict = pickle.load(fo, encoding='bytes')
	    return dict

	def __getitem__(self, index):
		fn, label = self._data[index]
		data = Image.open(self._action_groundtruth_path + fn).convert('RGB')
		if not self._transform is None:
			data = self._transform(data)

		return data, label

	def __len__(self):
		return len(self._data)


if __name__ == '__main__':
	model = resnet18(pretrained=False, num_classes=params['num_classes'])
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters())
	gpus = [0]
	if torch.cuda.is_available():
		# model = torch.nn.DataParallel(model, device_ids=gpus)
		model.cuda()
		pass

	train_data = Action_dataset(transform=transforms.ToTensor())
	test_data = Action_dataset(transform=transforms.ToTensor(), file_num=2)
	train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True)
	test_loader = DataLoader(dataset=test_data, batch_size=params['batch_size'], shuffle=False)
	dataiter = iter(train_loader)
	images, labels = dataiter.next()
	print(images.size())

	for epoch in range(params['num_epochs']):
	    since = time.time()
	    running_loss = 0
	    running_acc = 0
	    model.train()
	    for i, data in enumerate(train_loader, 1):
	        img, label = data
	        if torch.cuda.is_available():
	            img = img.cuda()
	            label = label.cuda()
	        out = model(img.float())
	        loss = criterion(out, label)
	        running_loss += loss.item()
	        _, pred = torch.max(out, 1)
	        running_acc += (pred==label).float().mean()

	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    running_acc /= i
	    running_loss /= i
	    print('time: ', time.time() - since)
	    print('epoch: ', epoch)
	    print('running_acc: ', running_acc)
	    print('running_loss: ', running_loss)
	model.eval()
	eval_loss = 0
	eval_acc = 0
	for i, data in enumerate(test_loader):
		img, label = data
		img = img.cuda()
		label = label.cuda()

		out = model(img.float())
		loss = criterion(out, label)

		eval_loss += loss.item()
		_, pred = torch.max(out, 1)
		eval_acc += (pred==label).float().mean()
	eval_acc /= i
	eval_loss /= i
	print('eval_acc: ', eval_acc)
	print('eval_loss: ', eval_loss)
