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
import random
import os
import sys
sys.path.append('..')

from common import *
import yaml
import pickle
from PIL import Image
import logging

f = open('config.yaml', 'r')
params = yaml.load(f, yaml.Loader)


class Action_dataset(torch.utils.data.Dataset):
	def __init__(self, action_groundtruth_path=params['action_file_path'], transform=None):
		super(Action_dataset, self).__init__()
		self._action_groundtruth_path = action_groundtruth_path
		# action_groundtruth = open(self._action_groundtruth_path + '/action.txt', 'r')
		self._img_path_list = []
		self._label_list = []
		# self._data = []
		self._index_pair_label = self.Go_through_text()
		print('self._index_pair_label: ', self._index_pair_label)
		# for line in action_groundtruth:
		# 	line = line.rstrip()
		# 	words = line.split()
		# 	self._data.append((words[0], int(words[1])))
		self._transform = transform

	def Go_through_text(self):
		action_groundtruth = open(self._action_groundtruth_path + '/action.txt', 'r')
		label_current = None
		shuffle_index_pair_label = []
		repeat_index_dict = {}
		pre_dict_index = 0
		for i, line in enumerate(action_groundtruth):
			line = line.rstrip()
			words = line.split()
			label = int(words[1])
			self._img_path_list.append(words[0])
			self._label_list.append(label)
			if label_current is None:
				label_current = label
				repeat_index_dict[i] = [1, label_current]
				pre_dict_index = i
			elif label_current == label:
				repeat_index_dict[pre_dict_index][0] += 1
			elif label_current != label:
				label_current = label
				repeat_index_dict[i] = [1, label_current]
				pre_dict_index = i
		print(repeat_index_dict)
		keys_shuffled = copy.deepcopy(list(repeat_index_dict.keys()))
		random.shuffle(keys_shuffled)
		for key in keys_shuffled:
			distance = random.randint(1, repeat_index_dict[key][0])
			if key + distance == len(self._img_path_list):
				continue
			shuffle_index_pair_label.append([key, key + distance, repeat_index_dict[key][1]])
			
		return shuffle_index_pair_label

	def __getitem__(self, index):
		# fn, label = self._data[index]
		# data = Image.open(self._action_groundtruth_path + fn).convert('RGB')
		index_pair_label = self._index_pair_label[index]
		current_img_path = self._img_path_list[index_pair_label[0]]
		future_img_path = self._img_path_list[index_pair_label[1]]
		label = index_pair_label[2]
		current_img = (np.array(Image.open(self._action_groundtruth_path + current_img_path).convert('RGB'))).transpose([1, 2, 0])
		future_img = (np.array(Image.open(self._action_groundtruth_path + future_img_path).convert('RGB'))).transpose([1, 2, 0])
		data = np.concatenate((current_img, future_img), axis=2)
		if not self._transform is None:
			data = self._transform(data)

		return data, label

	def __len__(self):
		# return len(self._data)
		return len(self._index_pair_label)


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
	# test_data = Action_dataset(transform=transforms.ToTensor(), file_num=2)
	train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True)
	# test_loader = DataLoader(dataset=test_data, batch_size=params['batch_size'], shuffle=False)
	dataiter = iter(train_loader)
	images, labels = dataiter.next()
	print(images.size())

	# exit()
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
