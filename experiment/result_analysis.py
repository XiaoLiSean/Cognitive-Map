from matplotlib import pyplot as plt
import csv
import numpy as np

if __name__ == '__main__':

	# for i in range(8):
	# 	for j in range(8):
	# 		print(i, j, np.sqrt((0.25 * i) ** 2 + (0.25 * j) ** 2))
	# exit()

	# distance_data = {}
	# coordinate_differences = []
	# coor_diff_corres_cases = []
	# heat_map_num = np.zeros((17, 17))

	# with open('distance.csv', 'r') as result:
	# # with open('normal.csv', 'r') as result:
	# 	reader = csv.reader(result)
	# 	action_num = 0
	# 	for row in reader:
	# 		distance = float(row[0])
	# 		if distance >= 2.25:
	# 			continue
	# 		action_num += 1

	# 		orientation = float(row[1])
	#     	# success = int(row[2])
	# 		coordinate_difference = [distance * np.cos(orientation * np.pi / 180), distance * np.sin(orientation * np.pi / 180)]
	# 		coordinate_difference[0] = int(np.round(coordinate_difference[0] / 0.25))
	# 		coordinate_difference[1] = int(np.round(coordinate_difference[1] / 0.25))

	# 		coor_diff_index = None
	# 		just_add = False
	# 		if not coordinate_difference in coordinate_differences:
	# 			coordinate_differences.append(coordinate_difference)
	# 			coor_diff_corres_cases.append(1)
	# 			just_add = True
	# 			coor_diff_index = coordinate_differences.index(coordinate_difference)
	# 		else:
	# 			coor_diff_index = coordinate_differences.index(coordinate_difference)
	# 			coor_diff_corres_cases[coor_diff_index] += 1


	# 		if not distance in list(distance_data.keys()):
	# 			distance_data[distance] = 1
	# 		else:
	# 			distance_data[distance] += 1

	# for index, coordinate_difference in enumerate(coordinate_differences):
	# 	heat_map_prob_index = [coordinate_difference[0], coordinate_difference[1]]
	# 	if np.abs(heat_map_prob_index[0]) > 8 or np.abs(heat_map_prob_index[1]) > 8:
	# 		continue
	# 	heat_map_prob_index[0] = -heat_map_prob_index[0]
	# 	heat_map_prob_index[1] = -heat_map_prob_index[1]
	# 	# if not heat_map_prob_index[0] == 1:
	# 	# 	continue

	# 	heat_map_prob_index[0] += 8
	# 	heat_map_prob_index[1] += 8

	# 	# heat_map_prob[heat_map_prob_index[0], heat_map_prob_index[1]] = round(coor_diff_corres_success_cases[index] / coor_diff_corres_cases[index], 3)
	# 	heat_map_num[heat_map_prob_index[0], heat_map_prob_index[1]] = coor_diff_corres_cases[index]
	# 	# num_test += coor_diff_corres_cases[index]

	# fig, ax = plt.subplots()
	# im = ax.imshow(heat_map_num)
	# print('action_num: ', action_num)

	# for i in range(17):
	#     for j in range(17):
	#         text = ax.text(j, i, str(int(heat_map_num[i, j])),
	#                        ha="center", va="center", color="w")

	# fig.tight_layout()
	# plt.show()

	# exit()


	# with open('distance_special_small.csv', 'r') as result:
	# # with open('normal.csv', 'r') as result:
	# 	reader = csv.reader(result)
	# 	action_num = 0
	# 	for row in reader:
	# 		distance = float(row[0])
	# 		if distance >= 2.25:
	# 			continue
	# 		action_num += 1
	# 		if not distance in list(distance_data.keys()):
	# 			distance_data[distance] = 1
	# 		else:
	# 			distance_data[distance] += 1

	# with open('distance_big.csv', 'r') as result:
	# # with open('normal.csv', 'r') as result:
	# 	reader = csv.reader(result)
	# 	action_num = 0
	# 	for row in reader:
	# 		distance = float(row[0])
	# 		if distance >= 2.25:
	# 			continue
	# 		action_num += 1
	# 		if not distance in list(distance_data.keys()):
	# 			distance_data[distance] = 1
	# 		else:
	# 			distance_data[distance] += 1

	# print('action_num: ', action_num)
	# distance_data_key = list(distance_data.keys())

	# distance_data_key.sort()

	# distance_data_value = []

	# for key in distance_data_key:
	# 	distance_data_value.append(distance_data[key])

	# plt.plot(distance_data_key, distance_data_value)
	# plt.title("distance_data")

	# for a, b in zip(distance_data_key, distance_data_value):
	# 	plt.text(a, b, b, ha='center', va='bottom', fontsize=20)

	# plt.show()

	# exit()

	compansation = 0.001

	heat_map_prob = np.zeros((17, 17))
	heat_map_num = np.zeros((17, 17))

	coordinate_differences = []
	coor_diff_corres_cases = []
	coor_diff_corres_success_cases = []
	# coordinate_difference_lookup = {'front': [1, 1], 'left': []}

	num_success = 0
	num_failure = 0

	success_distance = {}
	failure_distance = {}

	success_orientation = {}
	failure_orientation = {}

	with open('result.csv', 'r') as result:
	    reader = csv.reader(result)
	    for row in reader:
	    	distance = float(row[0])
	    	if distance > 2:
	    		continue
	    	orientation = row[1]
	    	success = int(row[2])
	    	coordinate_difference = [(distance + compansation) * np.cos(float(row[3]) * np.pi / 180),
	    							 (distance + compansation) * np.sin(float(row[3]) * np.pi / 180)]
	    	# print('coordinate_difference: ', coordinate_difference)
	    	# print('coordinate_difference[0] / 0.25: ', coordinate_difference[0] / 0.25)
	    	# print('coordinate_difference[1] / 0.25: ', coordinate_difference[1] / 0.25)
	    	coordinate_difference[0] = int(np.round(coordinate_difference[0] / 0.25))
	    	coordinate_difference[1] = int(np.round(coordinate_difference[1] / 0.25))
	    	# print('coordinate_difference: ', coordinate_difference)
	    	# print('float(row[3]): ', float(row[3]))
	    	# print('----------------------------')
	    	coor_diff_index = None
	    	just_add = False
	    	if not coordinate_difference in coordinate_differences:
	    		coordinate_differences.append(coordinate_difference)
	    		coor_diff_corres_cases.append(1)
	    		coor_diff_corres_success_cases.append(0)
	    		just_add = True
	    		coor_diff_index = coordinate_differences.index(coordinate_difference)
	    	else:
	    		coor_diff_index = coordinate_differences.index(coordinate_difference)
	    		coor_diff_corres_cases[coor_diff_index] += 1
    		
	    	if success == 1:
	    		num_success += 1
	    		coor_diff_corres_success_cases[coor_diff_index] += 1

	    		if not distance in list(success_distance.keys()):
	    			success_distance[distance] = 1
	    		else:
	    			success_distance[distance] += 1
	    		if not orientation in list(success_orientation.keys()):
	    			success_orientation[orientation] = 1
	    		else:
	    			success_orientation[orientation] += 1
	    	else:
	    		num_failure += 1
	    		if not distance in list(failure_distance.keys()):
	    			failure_distance[distance] = 1
	    		else:
	    			failure_distance[distance] += 1
	    		if not orientation in list(failure_orientation.keys()):
	    			failure_orientation[orientation] = 1
	    		else:
	    			failure_orientation[orientation] += 1

	success_distance_key = list(success_distance.keys())
	failure_distance_key = list(failure_distance.keys())

	success_distance_key.sort()
	print(success_distance_key)
	failure_distance_key.sort()

	success_distance_value = []
	failure_distance_value = []
	all_distance_value = []

	success_rate_vs_distance = []
	for key in success_distance_key:
		success_distance_value.append(success_distance[key])
		if key in list(failure_distance.keys()):
			all_distance_value.append(success_distance[key] + failure_distance[key])
		else:
			all_distance_value.append(success_distance[key])
	for key in failure_distance_key:
		failure_distance_value.append(failure_distance[key])
		if not key in list(success_distance.keys()):
			success_rate_vs_distance.append(0)
		elif not key in list(failure_distance.keys()):
			success_rate_vs_distance.append(1)
		else:
			success_rate_vs_distance.append(success_distance[key] / (success_distance[key] + failure_distance[key]))


	num_test = 0
	for index, coordinate_difference in enumerate(coordinate_differences):
		heat_map_prob_index = [coordinate_difference[0], coordinate_difference[1]]
		if np.abs(heat_map_prob_index[0]) > 8 or np.abs(heat_map_prob_index[1]) > 8:
			continue
		heat_map_prob_index[0] = -heat_map_prob_index[0]
		heat_map_prob_index[1] = -heat_map_prob_index[1]
		# if not heat_map_prob_index[0] == 1:
		# 	continue

		heat_map_prob_index[0] += 8
		heat_map_prob_index[1] += 8

		heat_map_prob[heat_map_prob_index[0], heat_map_prob_index[1]] = round(coor_diff_corres_success_cases[index] / coor_diff_corres_cases[index], 3)
		heat_map_num[heat_map_prob_index[0], heat_map_prob_index[1]] = coor_diff_corres_cases[index]
		num_test += coor_diff_corres_cases[index]

	print('coordinate_differences: ', coordinate_differences)
	print('success_rate: ', num_success / (num_success + num_failure))
	print('total num: ', (num_success + num_failure))
	print('num_test: ', num_test)

	fig, ax = plt.subplots()
	im = ax.imshow(heat_map_prob)

	for i in range(17):
	    for j in range(17):
	        text = ax.text(j, i, str(heat_map_prob[i, j]),
	                       ha="center", va="center", color="w")

	fig.tight_layout()
	plt.show()

	fig, ax = plt.subplots()
	im = ax.imshow(heat_map_num)

	for i in range(17):
	    for j in range(17):
	        text = ax.text(j, i, str(int(heat_map_num[i, j])),
	                       ha="center", va="center", color="w")

	fig.tight_layout()
	plt.show()

	exit()

	plt.figure(12)
	# plt.subplot(221)
	# # plt.plot(list(success_distance.keys()), list(success_distance.values()))
	# plt.plot(success_distance_key, success_distance_value)
	# plt.title("success_case_vs_distance")
	# # plt.show()

	# plt.subplot(222)
	# # plt.plot(list(failure_distance.keys()), list(failure_distance.values()))
	# plt.plot(failure_distance_key, failure_distance_value)
	# plt.title("failure_case_vs_distance")
	# plt.show()

	plt.subplot(211)
	# plt.plot(list(failure_distance.keys()), list(failure_distance.values()))
	plt.plot(success_distance_key, all_distance_value, marker='o')
	plt.title("case_number_vs_distance")

	for a, b in zip(failure_distance_key, all_distance_value):
		plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=10)

	plt.subplot(212)
	# plt.plot(list(failure_distance.keys()), list(failure_distance.values()))
	plt.plot(failure_distance_key, success_rate_vs_distance, marker='o')
	plt.title("success_rate_vs_distance")
	for a, b in zip(failure_distance_key, success_rate_vs_distance):
		plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=10)

	# plt.subplot(223)
	# plt.plot(list(success_orientation.keys()), list(success_orientation.values()))
	# plt.title("success_orientation")
	# # plt.show()

	# plt.subplot(224)
	# plt.plot(list(failure_orientation.keys()), list(failure_orientation.values()))
	# plt.title("failure_orientation")
	plt.show()