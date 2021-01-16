import os
import sys


if __name__ == '__main__':
	print(os.getcwd())
	# action_groundtruth = open('data/test_dataset/action.txt', 'r')
	action_groundtruth = open('data/action.txt', 'r')

	labels = []
	file_name = []
	for i, line in enumerate(action_groundtruth):
		line = line.rstrip()
		words = line.split()
		labels.append(int(words[1]))
		file_name.append(words[0])
	
	# action_label_text_file = open('data/test_dataset/action_fix.txt', 'w')
	action_label_text_file = open('data/action_fix.txt', 'w')
	for i in range(len(file_name) - 1):
		action_label_text_file.write(file_name[i] + ' ' + str(labels[i + 1]) + '\n')

