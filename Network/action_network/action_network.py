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
import cv2
from skimage.transform import resize

# print(os.getcwd())
current_path = os.getcwd()
print(current_path)
print(type(current_path.find('Cognitive-Map') + len('Cognitive-Map/')))
initial_path = current_path[0:current_path.find('Cognitive-Map') + len('Cognitive-Map/')]
print(initial_path)
if not initial_path[-1] == '/':
	initial_path = initial_path + '/'
action_network_path = initial_path + 'Network/action_network/'
sys.path.append(initial_path + 'Network/action_network')

f = open(action_network_path + 'config/config.yaml', 'r')
params = yaml.load(f, yaml.Loader)

class Action_dataset(torch.utils.data.Dataset):
	def __init__(self, action_groundtruth_path, transform=None):
		super(Action_dataset, self).__init__()
		self._action_groundtruth_path = action_groundtruth_path
		# action_groundtruth = open(self._action_groundtruth_path + '/action.txt', 'r')
		self._img_path_list = []
		self._label_list = []
		# self._index_pair_label = self.Go_through_text()
		self._index_pair_label = self.SPTM_like_go_through_text()
		self._transform = transform

	def SPTM_like_go_through_text(self):
		action_groundtruth = open(self._action_groundtruth_path + '/action_fix.txt', 'r')
		label_current = None
		starting_image_index = None
		end_image_index = None
		action_index = None
		shuffle_index_pair_label = []
		action_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
		for i, line in enumerate(action_groundtruth):
			line = line.rstrip()
			words = line.split()
			label = int(words[1])
			add = True
			self._img_path_list.append(words[0])
			self._label_list.append(label)


			if starting_image_index is None:
				starting_image_index = i
				action_index = label

			if action_index in [1, 2]:
				add = random.choice([True, True, False])

			if label < 0:
				end_image_index = i
				if end_image_index == starting_image_index:
					starting_image_index = None
					continue
				if add:
					shuffle_index_pair_label.append([starting_image_index, end_image_index, action_index])
					action_num[action_index] += 1


				starting_image_index = None
		# print(shuffle_index_pair_label)
		return shuffle_index_pair_label

	def Go_through_text(self):
		action_groundtruth = open(self._action_groundtruth_path + '/action_fix.txt', 'r')
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
		# print(repeat_index_dict)
		keys_shuffled = copy.deepcopy(list(repeat_index_dict.keys()))
		# random.shuffle(keys_shuffled)
		for key in keys_shuffled:
			if repeat_index_dict[key][1] < 0:
				continue
			starting_key = copy.deepcopy(key)
			frame_left = copy.deepcopy(repeat_index_dict[key][0])
			while starting_key < key + repeat_index_dict[key][0]:
				rand_frame_distance = random.randint(1, min(frame_left, 3))
				# if starting_key + rand_frame_distance in keys_shuffled:
				# 	if repeat_index_dict[starting_key + rand_frame_distance][1] < 0:
				# 		break
				# print('rand_frame_distance: ', rand_frame_distance)
				# print('starting_key: ', starting_key)
				# print('shuffle_index_pair_label: ', shuffle_index_pair_label)
				if starting_key + rand_frame_distance == len(self._img_path_list):
					break
				shuffle_index_pair_label.append([starting_key, starting_key + rand_frame_distance, repeat_index_dict[key][1]])
				starting_key = starting_key + rand_frame_distance
				frame_left = frame_left - rand_frame_distance
		# print('---------------------------')
		# print(shuffle_index_pair_label)
		return shuffle_index_pair_label

	def __getitem__(self, index):
		# fn, label = self._data[index]
		# data = Image.open(self._action_groundtruth_path + fn).convert('RGB')
		index_pair_label = self._index_pair_label[index]
		current_img_path = self._img_path_list[index_pair_label[0]]
		future_img_path = self._img_path_list[index_pair_label[1]]
		label = index_pair_label[2]
		current_img = (np.array(Image.open(self._action_groundtruth_path + current_img_path).convert('RGB'))).transpose([0, 1, 2])
		# print('current_img: ', current_img.shape)
		future_img = (np.array(Image.open(self._action_groundtruth_path + future_img_path).convert('RGB'))).transpose([0, 1, 2])
		data = np.concatenate((current_img, future_img), axis=1)
		# print('data shape', data.shape)
		if not self._transform is None:
			data = self._transform(data)
		# print(data.size())
		return data, label

	def __len__(self):
		# return len(self._data)
		return len(self._index_pair_label)


class Action_network():
	def __init__(self, num_classes=6, weight_file_path=action_network_path + 'weight/ithor.pkl'):
		self._model = resnet18(pretrained=False, num_classes=num_classes)
		print('torch.cuda.is_available(): ', torch.cuda.is_available())
		if torch.cuda.is_available():
			self._model.cuda()
		self._model.load_state_dict(torch.load(weight_file_path))
		self._model.eval()

	def predict(self, image_current, image_goal):

		image_current_resize = np.resize(resize(image_current, (300, 300, 3)), (300, 300, 3))
		image_goal_resize = np.resize(resize(image_goal, (300, 300, 3)), (300, 300, 3))

		img = np.concatenate((image_current_resize, image_goal_resize), axis=1)
		# img = np.concatenate((image_current, image_goal), axis=1)


		img = transforms.ToTensor()(img)
		img_shape = img.shape
		img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

		if torch.cuda.is_available():
			img = img.cuda()
		out = self._model(img.float())
		# print(out)
		_, pred = torch.max(out, 1)
		return pred

	def predict_fuse(self, img_fused):
		# img = np.concatenate((image_current, image_goal), axis=1)
		# img = transforms.ToTensor()(img)
		img_shape = img_fused.shape
		img = copy.deepcopy(img_fused)
		img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
		# print('img: ', img)
		if torch.cuda.is_available():
			img = img.cuda()
		out = self._model(img.float())
		# print(out)
		_, pred = torch.max(out, 1)
		return pred


if __name__ == '__main__':
	model = resnet18(pretrained=True)
	# model = resnet18(pretrained=False, num_classes=params['num_classes'])
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, params['num_classes'])

	# model.load_state_dict(torch.load('weight/params_SPTM_like_back_left_right_large_rot_90_new_long_more_more_left_special.pkl'))
	# exit()
	# model.load_state_dict(torch.load('weight/params_SPTM_like_back_left_right_large.pkl'))

	# model.load_state_dict(torch.load('params.pkl'))
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters())
	gpus = [0]
	if torch.cuda.is_available():
		# model = torch.nn.DataParallel(model, device_ids=gpus)
		model.cuda()
		pass

	train_data = Action_dataset(transform=transforms.ToTensor(), action_groundtruth_path=params['action_file_path'])
	# test_data = Action_dataset(transform=transforms.ToTensor(), action_groundtruth_path=params['test_action_file_path'])
	train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True)
	# test_loader = DataLoader(dataset=test_data, batch_size=int(params['batch_size'] / 2), shuffle=False)
	# dataiter = iter(train_loader)
	print(len(train_data))
	# print(len(test_data))
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
	        # print('(pred==label).float().mean(): ', (pred==label).float().mean())
	        # print('pred: ', pred)
	        # print('label: ', label)
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    running_acc /= i
	    running_loss /= i
	    print('time: ', time.time() - since)
	    print('epoch: ', epoch)
	    print('running_acc: ', running_acc.item())
	    print('running_loss: ', running_loss)
	    torch.save(model.state_dict(), 'params.pkl')
	    if running_acc.item() > 0.98:
	    	break
	torch.save(model.state_dict(), 'params.pkl')
	exit()
	# model.load_state_dict(torch.load('params_SPTM_like_backward_large.pkl'))
	model.eval()
	eval_loss = 0
	eval_acc = 0

	# exit()

	for i, data in enumerate(test_loader):
		img, label = data
		img = img.cuda()
		label = label.cuda()

		out = model(img.float())
		loss = criterion(out, label)

		eval_loss += loss.item()
		_, pred = torch.max(out, 1)
		eval_acc += (pred==label).float().mean()
		# print((pred==label).float().mean().item())
		# print(i)
	eval_acc /= (i + 1)
	eval_loss /= i
	print(i)
	print('eval_acc: ', eval_acc)
	print('eval_loss: ', eval_loss)
