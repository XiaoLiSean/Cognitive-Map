import copy
import random
import logging
import os
import sys
import numpy as np


class SFrelement():
	def __init__(self):
		self.amplitude = 0
		self.phase = 0
		self.frequency = 0


class subnode_dynamic_record():
	def __init__(self):
		self._subnode_name = None
		self._object_name = []
		self._times = []
		self._signals = []


class Frelement():
	def __init__(self, time_unit='min'):
		self._num_periodicity = 24
		self._order = -1
		self._measurement_num = 0
		self._firstTime = -1
		self._lastTime = -1
		self._lastMeasurement = None
		self._gain = 0.5
		self._SFrelements = []
		self._periods = [(24*60)/(i+1) for i in range(0, self._num_periodicity)]
		self._time_unit = time_unit
		if self._time_unit == 'sec':
			self._period = 24 * 3600
		elif self._time_unit == 'min':
			self._period = 24 * 60
		elif self._time_unit == 'hour':
			self._period = 24

		self._outlierSet = []
		self._outlierNum = len(self._outlierSet)

	# Add new mearsurement/observation, add to outlier set if not fitting the model
	def add(self, time, value):

		if not self.retrieve(time=time) == value and not time in self._outlierSet:
			self._outlierSet.append(time)
			self._outlierNum = len(self._outlierSet)
		if self._firstTime == -1:
			self._firstTime = time
			self._lastTime = time
		if time > self._lastTime:
			self._lastTime = time

	# retrieve the recorded observation based on current model and outlierSet (p > 0.5) XOR (time in outlier set)
	def retrieve(self, time):
		retrieved_bool = bool(self.estimate(time_estimate=time) > 0.5) ^ bool((time in self._outlierSet))
		if retrieved_bool:
			return 1
		else:
			return 0

	# reconstruct the groundtruth observations at [times]
	def reconstruct(self, times):

		reconstructed_signal = []
		for time_observed in times:
			reconstructed_signal.append(self.retrieve(time=time_observed))

		return reconstructed_signal

	# build the model
	def build(self, times, signals, length, orderi):
		
		numFrequencies = self._num_periodicity
		frequencies = [float(self._period)/(i+1) for i in range(numFrequencies)]
		real = [0 for _ in range(numFrequencies)]
		imag = [0 for _ in range(numFrequencies)]

		signalLength = length
		
		self._order = orderi
		self._SFrelements = [SFrelement() for _ in range(self._order)]
		self._gain = 0
		self._firstTime = min(times)
		self._lastTime = max(times)

		for i in range(length):
			self._gain += signals[i]

		self._gain /= signalLength
		balance = self._gain

		for j in range(length):
			for i in range(numFrequencies):
				real[i] += (signals[j] - balance) * np.cos(2 * np.pi * float(times[j] / frequencies[i]))
				imag[i] += (signals[j] - balance) * np.sin(2 * np.pi * float(times[j] / frequencies[i]))

		tmpFrelements = [SFrelement() for _ in range(numFrequencies)]
		for i in range(numFrequencies):
			tmpFrelements[i].amplitude = real[i] ** 2 + imag[i] ** 2
			tmpFrelements[i].frequency = i

		SFrelements_sorted = sorted(tmpFrelements, key=lambda x: x.amplitude, reverse=True)

		for i in range(self._order):

			index = int(SFrelements_sorted[i].frequency)
			self._SFrelements[i].amplitude = np.sqrt(SFrelements_sorted[i].amplitude) / signalLength
			self._SFrelements[i].phase = np.arctan2(imag[index], real[index])
			self._SFrelements[i].frequency = frequencies[index]

		self._outlierSet = []
		self._outlierNum = len(self._outlierSet)
		reconstructed_signal = self.reconstruct(times=times)

		for signal_index, signal in enumerate(signals):
			if not signal == reconstructed_signal[signal_index]:
				self._outlierSet.append(times[signal_index])

		self._outlierNum = len(self._outlierSet)

	# Updte the model when the number of outlier is large
	def update(self, orderi):

		times = list(range(self._firstTime, self._lastTime + 1))
		reconstructed_signal = self.reconstruct(times=times)
		self.build(times=times, signals=reconstructed_signal, length=len(times), orderi=orderi)

	# estimate the probability of state at certain time with current model
	def estimate(self, time_estimate):

		estimate = self._gain
		for i in range(self._order):
			estimate += 2 * self._SFrelements[i].amplitude * np.cos(time_estimate / self._SFrelements[i].frequency * 2 * np.pi - self._SFrelements[i].phase)

		return estimate

	def evaluate(self, times, signal, length):

		error = 0
		samples_num = length

		for i in range(length):
			error += np.abs(self.estimate(time_estimate=times[i]) - signal[i])

		return error / samples_num


if __name__ == '__main__':
	test_Frelement = Frelement()

	test_part = 3
	times_test = [i for i in range(int(test_Frelement._period))]
	states = [1 for _ in range(len(times_test))]

	# for i in range(int(test_Frelement._period / test_part)):
	# 	states[i] = 0

	# times_test = [1, test_Frelement._period]
	# states = [1, 1]
	print('states: ', states)
	test_Frelement.build(times=times_test, signals=states, length=len(states), orderi=12)
	# print('first_time: ',test_Frelement._firstTime)
	# print('first_time: ',test_Frelement._lastTime)

	# print(test_Frelement.estimate(time_estimate=7))
	# exit()

	print(test_Frelement._outlierSet)
	for i in range(int(test_Frelement._period / test_part)):
		test_Frelement.add(time=i, value=0)
	test_Frelement.add(time=1, value=0)
	test_Frelement.add(time=2, value=0)
	test_Frelement.add(time=3, value=0)
	test_Frelement.update(orderi=12)
	print(test_Frelement._outlierSet)

	exit()

	print(test_Frelement.estimate(time_estimate=10))
	print(test_Frelement.estimate(time_estimate=test_Frelement._period + 10))
	# error = 0
	# for i in range(test_Frelement._period):
	# 	estimate = test_Frelement.estimate(time_estimate=i)
	# 	# print(estimate)
	# 	if i < test_Frelement._period / test_part:
	# 		if not estimate > 0.5:
	# 			error += 1
	# 	else:
	# 		if not estimate < 0.5:
	# 			error += 1
	# print('error: ', error)
	# times_test = [i for i in range(int(test_Frelement._period))]
	# states = [0 for _ in range(len(times_test))]
	# states[]
	print(test_Frelement.evaluate(times=times_test, signal=states, length=len(states)))
	exit()
