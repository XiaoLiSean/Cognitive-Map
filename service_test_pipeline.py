import argparse, copy
import multiprocessing, time
from lib.navigation import Navigation
from Map.map_plotter import Plotter
from distutils.util import strtobool
import os, csv, itertools
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--scene_type", type=int, default=1,  help="Choose scene type for simulation, 1 for Kitchens, 2 for Living rooms, 3 for Bedrooms, 4 for Bathrooms")
parser.add_argument("--scene_num", type=int, default=26,  help="Choose scene num for simulation, from 1 - 30")
parser.add_argument("--grid_size", type=float, default=0.25,  help="Grid size of AI2THOR simulation")
parser.add_argument("--rotation_step", type=float, default=10,  help="Rotation step of AI2THOR simulation")
parser.add_argument("--sleep_time", type=float, default=0.005,  help="Sleep time between two actions")
parser.add_argument("--save_directory", type=str, default='./data',  help="Data saving directory")
parser.add_argument("--overwrite_data", type=lambda x: bool(strtobool(x)), default=False, help="overwrite the existing data or not")
parser.add_argument("--log_level", type=int, default=2,  help="Level of showing log 1-5 where 5 is most detailed")
parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False,  help="Output debug info if True")
parser.add_argument("--test_data", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting test dataset")
parser.add_argument("--special", type=lambda x: bool(strtobool(x)), default=False, help="True for collecting special long range dataset")
parser.add_argument("--AI2THOR", type=lambda x: bool(strtobool(x)), default=False, help="True for RobotTHOR false for ITHOR")
args = parser.parse_args()

def preprocessData():
	# get tested scene name
	tags = []
	tags_order = {}
	for i in range(1, 5):
		for j in range(26, 31):
			if i == 1:
				scene_name = 'FloorPlan'+str(j)
			else:
				scene_name = 'FloorPlan'+str(i)+str(j)
			tags_order.update({scene_name: len(tags)})
			tags.append(scene_name)
	# read all available test files
	data_dir = './Network/service_test'
	processedData = {}
	for file in os.listdir(data_dir):
		netName = file.split('.')[0]
		processedData.update({netName:
							  dict(neighbor_success=np.zeros(len(tags)), neighbor_navi_fail=np.zeros(len(tags)),
							  	   neighbor_loca_fail=np.zeros(len(tags)), neighbor_collision=np.zeros(len(tags)),
							  	   success=np.zeros(len(tags)), navi_fail=np.zeros(len(tags)),
								   loca_fail=np.zeros(len(tags)), collision=np.zeros(len(tags)),
								   neighbor_case_num=np.zeros(len(tags)), case_num=np.zeros(len(tags)))})
		filename = data_dir + '/' + file
		# read data from individual file
		with open(filename, newline='') as csvfile:
			spamreader = csv.reader(csvfile)
			for row in spamreader:
				sceneName = row[0]
				idx = tags_order[sceneName]
				processedData[netName]['neighbor_case_num'][idx] = float(row[3])
				processedData[netName]['neighbor_success'][idx] = 1- float(row[4])/float(row[3])
				processedData[netName]['neighbor_navi_fail'][idx] = float(row[5])/float(row[3])
				processedData[netName]['neighbor_loca_fail'][idx] = float(row[6])/float(row[3])
				processedData[netName]['neighbor_collision'][idx] = float(row[7])/float(row[3])
				processedData[netName]['case_num'][idx] = float(row[1])
				processedData[netName]['success'][idx] = 1 - float(row[2])/float(row[1])
				processedData[netName]['navi_fail'][idx] = float(row[8])/float(row[1])
				processedData[netName]['loca_fail'][idx] = float(row[9])/float(row[1])
				processedData[netName]['collision'][idx] = float(row[10])/float(row[1])
	return processedData, tags

def plot_statistics():
	data, tags = preprocessData()
	fig1, ax1 = plt.subplots(figsize=(6,5))
	fig2, ax2 = plt.subplots(figsize=(6,5))
	num = len(data)
	# colors = ['#5A90BE', '#5C9CC0', '#A18DA8', '#7B5B42', '#CBB8BA']
	colors = ['purple', 'green', 'orange', 'blue', 'brown']
	density = 3
	# patterns = ['+'*density, '.'*density, '\\'*density, "/"*density]
	patterns = [None, None, None, None]
	alphas = [1.0, 0.6, 0.3, 0.1]
	for i, net in enumerate(data):
		ax1.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['neighbor_success'], width=1, color=colors[i], hatch=patterns[0], alpha=alphas[0], edgecolor='white')
		ax1.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['neighbor_loca_fail'], width=1, color=colors[i], hatch=patterns[1], alpha=alphas[1], edgecolor='white', bottom=data[net]['neighbor_success'])
		ax1.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['neighbor_navi_fail'], width=1, color=colors[i], hatch=patterns[2], alpha=alphas[2], edgecolor='white', bottom=data[net]['neighbor_success']+data[net]['neighbor_loca_fail'])
		ax1.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['neighbor_collision'], width=1, color=colors[i], hatch=patterns[3], alpha=alphas[3], edgecolor='white', bottom=data[net]['neighbor_success']+data[net]['neighbor_loca_fail']+data[net]['neighbor_navi_fail'])
		ax2.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['success'], width=1, color=colors[i], hatch=patterns[0], alpha=alphas[0], edgecolor='white')
		ax2.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['loca_fail'], width=1, color=colors[i], hatch=patterns[1], alpha=alphas[1], edgecolor='white', bottom=data[net]['success'])
		ax2.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['navi_fail'], width=1, color=colors[i], hatch=patterns[2], alpha=alphas[2], edgecolor='white', bottom=data[net]['success']+data[net]['loca_fail'])
		ax2.bar(np.arange(len(tags))*(num+1)-(num-i), data[net]['collision'], width=1, color=colors[i], hatch=patterns[3], alpha=alphas[3], edgecolor='white', bottom=data[net]['success']+data[net]['loca_fail']+data[net]['navi_fail'])

	ax1.set_xticks(np.arange(len(tags))*(num+1)-num/2.0)
	ax1.set_xticklabels(tags, rotation=90)
	ax1.set_yticks(np.arange(11)/10)
	ax1.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
	#ax1.grid(True)
	fig1.tight_layout()

	ax2.set_xticks(np.arange(len(tags))*(num+1)-num/2.0)
	ax2.set_xticklabels(tags, rotation=90)
	ax2.set_yticks(np.arange(11)/10)
	ax2.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
	#ax2.grid(True)
	fig1.tight_layout()

	plt.show()

def plot_single_net_testing():
	data, tags = preprocessData()
	fig1, ax1 = plt.subplots(figsize=(6,5))
	# colors = ['#5A90BE', '#5C9CC0', '#A18DA8', '#7B5B42', '#CBB8BA']
	colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
	# colors = ['purple', 'green', 'orange', 'blue', 'brown']
	# density = 3
	# patterns = ['+'*density, '.'*density, '\\'*density, "/"*density]
	patterns = [None, None, None, None]
	alphas = [0.7, 1.0]
	num = 2 # number of bars in a single scene plotted in ax1
	for net in data:
		if net == 'rnet':
			ax1.bar(np.arange(len(tags))*(num+1)-(num), data[net]['neighbor_success'], width=1, color=colors[0], alpha=alphas[0], edgecolor='white',
				label='Success (neighbor, {:,.1%})'.format(np.dot(data[net]['neighbor_success'], data[net]['neighbor_case_num']) / np.sum(data[net]['neighbor_case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num-1), data[net]['success'], width=1, color=colors[0], alpha=alphas[1], edgecolor='white',
				label='Success (arbitrary, {:,.1%})'.format(np.dot(data[net]['success'], data[net]['case_num']) / np.sum(data[net]['case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num), data[net]['neighbor_loca_fail'], width=1, color=colors[1], alpha=alphas[0], bottom=data[net]['neighbor_success'],
				label='Localization fail (neighbor, {:,.1%})'.format(np.dot(data[net]['neighbor_loca_fail'], data[net]['neighbor_case_num']) / np.sum(data[net]['neighbor_case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num-1), data[net]['loca_fail'], width=1, color=colors[1], alpha=alphas[1], edgecolor='white', bottom=data[net]['success'],
				label='Localization fail (arbitrary, {:,.1%})'.format(np.dot(data[net]['loca_fail'], data[net]['case_num']) / np.sum(data[net]['case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num), data[net]['neighbor_navi_fail'], width=1, color=colors[2], alpha=alphas[0], bottom=data[net]['neighbor_success']+data[net]['neighbor_loca_fail'],
				label='Navigation fail (neighbor, {:,.1%})'.format(np.dot(data[net]['neighbor_navi_fail'], data[net]['neighbor_case_num']) / np.sum(data[net]['neighbor_case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num-1), data[net]['navi_fail'], width=1, color=colors[2], alpha=alphas[1], edgecolor='white', bottom=data[net]['success']+data[net]['loca_fail'],
				label='Navigation fail (arbitrary, {:,.1%})'.format(np.dot(data[net]['navi_fail'], data[net]['case_num']) / np.sum(data[net]['case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num), data[net]['neighbor_collision'], width=1,  color=colors[3], alpha=alphas[0], bottom=data[net]['neighbor_success']+data[net]['neighbor_loca_fail']+data[net]['neighbor_navi_fail'],
				label='Collision (neighbor, {:,.1%})'.format(np.dot(data[net]['neighbor_collision'], data[net]['neighbor_case_num']) / np.sum(data[net]['neighbor_case_num'])))
			ax1.bar(np.arange(len(tags))*(num+1)-(num-1), data[net]['collision'], width=1, color=colors[3], alpha=alphas[1], edgecolor='white', bottom=data[net]['success']+data[net]['loca_fail']+data[net]['navi_fail'],
				label='Collision (arbitrary, {:,.1%})'.format(np.dot(data[net]['collision'], data[net]['case_num']) / np.sum(data[net]['case_num'])))
		else:
			pass

	ax1.set_xticks(np.arange(len(tags))*(num+1)-num/2.0)
	ax1.set_xticklabels(tags, rotation=90)
	ax1.set_yticks(np.arange(11)/10)
	ax1.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(11)/10])
	ax1.set_ylabel('Case Percent', rotation=90)
	ax1.grid(True)
	ax1.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=4)
	fig1.tight_layout()

	plt.show()

if __name__ == '__main__':
	'''netName \in ['rnet', 'resnet50', 'vgg16', 'googlenet', 'resnext50_32x4d']'''
	navigation = Navigation(netName='resnet50', scene_type=args.scene_type, scene_num=args.scene_num, save_directory=args.save_directory, AI2THOR=args.AI2THOR)
	navigation.Update_node_generator()
	navigation.Update_topo_map_env()
	navigation.Update_planner_env()
	navigation.node_generator.Shuffle_scene()
	navigation.nav_test_simplified()
	# plot_single_net_testing()
