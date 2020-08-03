import os
import torch
from variantworks.sample_encoder import SummaryEncoder
from test_utils import get_data_folder
import numpy as np

class TestRegion(object):
	def __init__(self):
		self.start_pos = 0
		self.end_pos = 14460
		self.pileup = os.path.join(get_data_folder(), "subreads_and_truth.pileup")

def test_counts_correctness():
	region = TestRegion()
	encoder = SummaryEncoder(training=False)
	pileup_counts, positions = encoder(region)
	correct_counts = np.load(os.path.join(get_data_folder(), "sample_counts.npy"))
	assert(pileup_counts.shape == correct_counts.shape)
	assert(np.allclose(pileup_counts, correct_counts))

def test_positions_correctness():
	region = TestRegion()
	encoder = SummaryEncoder(training=False)
	pileup_counts, positions = encoder(region)
	correct_positions = np.load(os.path.join(get_data_folder(), "sample_positions.npy"))
	assert(positions.shape == correct_positions.shape)
	all_equal = True
	for i in range(len(positions)):
		if (positions[i] != correct_positions[i]):
			all_equal = False
			break
	assert(all_equal)

if __name__ == '__main__':
	test_counts_correctness()
	test_positions_correctness()



