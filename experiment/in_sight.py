from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from dijkstar import Graph, find_path
from distutils.util import strtobool
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import time
import copy
import argparse
import random
import logging
import os
import sys


controller = Controller(scene='FloorPlan301', gridSize=0.25, fieldOfView=120, renderObjectImage=True)
event = controller.step('Pass')
frame = copy.deepcopy(event.frame)
things = list(event.instance_detections2D.keys())

object_test = event.metadata['objects'][10]
points = object_test['objectOrientedBoundingBox']

all_objects = copy.deepcopy(event.metadata['objects'])
agent_position = np.array(list(event.metadata['agent']['position'].values()))

axis = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
area_list = {}
area_should_occupy = {}
for objects in all_objects:
	object_area = 0
	pos_side_pts = {'x': [], 'y': [], 'z': []}
	neg_side_pts = {'x': [], 'y': [], 'z': []}
	if objects['objectOrientedBoundingBox'] is None:
		bbox_points = objects['axisAlignedBoundingBox']['cornerPoints']
	else:
		bbox_points = objects['objectOrientedBoundingBox']['cornerPoints']
	bbox_center = np.sum(bbox_points, axis=0) / 8
	agent_bbox_vec = bbox_center - agent_position
	agent_bbox_vec = agent_bbox_vec / np.linalg.norm(agent_bbox_vec)
	orientation = objects['rotation']
	orientation_R = R.from_euler('zxy', [ orientation['z'], orientation['x'], orientation['y']], degrees=True)
	axis_rot = {}
	for key in list(axis.keys()):
		axis_rot[key] = np.array(orientation_R.apply(axis[key]))
	for i in range(len(bbox_points)):
		for j in range(len(bbox_points)):
			if i == j:
				continue
			dir_vec = np.array(list(map(lambda x, y: x - y, bbox_points[i], bbox_points[j])))
			dir_vec = dir_vec / np.linalg.norm(dir_vec)
			for key in list(axis_rot.keys()):
				if np.linalg.norm(dir_vec - axis_rot[key]) < 0.02:
					pos_side_pts[key].append(bbox_points[i])
					neg_side_pts[key].append(bbox_points[j])
	for key in list(pos_side_pts.keys()):
		pos_center = np.sum(pos_side_pts[key], axis=0) / 4
		neg_center = np.sum(neg_side_pts[key], axis=0) / 4
		line_center_length = []
		min_dis = 100
		min_dis_index = -1
		for i in range(3):
			line_center_length.append(np.linalg.norm((np.array(pos_side_pts[key][i + 1]) + np.array(pos_side_pts[key][0])) / 2 - pos_center))
			if np.linalg.norm((np.array(pos_side_pts[key][i + 1]) + np.array(pos_side_pts[key][0])) / 2 - pos_center) < min_dis:
				min_dis = np.linalg.norm((np.array(pos_side_pts[key][i + 1]) + np.array(pos_side_pts[key][0])) / 2 - pos_center)
				min_dis_index = i

		line_center_length.remove(line_center_length[min_dis_index])
		area_surface = line_center_length[0] * line_center_length[1] * 4
		pos_vector = pos_center - bbox_center
		neg_vector = neg_center - bbox_center
		pos_vector = pos_vector / np.linalg.norm(pos_vector)
		neg_vector = neg_vector / np.linalg.norm(neg_vector)
		object_area += max(np.dot(pos_vector, agent_bbox_vec) * area_surface, 0)
		object_area += max(np.dot(neg_vector, agent_bbox_vec) * area_surface, 0)
		
	area_list[objects['objectId']] = object_area
	area_should_occupy[objects['objectId']] = object_area / (np.pi * (objects['distance'] * np.tan(120 / 2 * np.pi / 180)) ** 2) * 90000


pixel_predeict_vs_mask ={}
in_sight = {}
for key in list(area_should_occupy.keys()):
	if key in list(event.instance_masks.keys()):
		pixel_predeict_vs_mask[key] = [area_should_occupy[key], sum(sum(event.instance_masks[key]))]
		if pixel_predeict_vs_mask[key][1] / pixel_predeict_vs_mask[key][0] < 0.01:
			in_sight[key] = False
		else:
			in_sight[key] = True
