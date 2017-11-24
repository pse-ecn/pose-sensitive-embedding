import argparse
import glob
import sys

import numpy as np
from scipy.misc import imread


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input-pattern', help='input-pattern')
	args = parser.parse_args()

	all_images = glob.glob(args.input_pattern)

	calculate_mean_std(all_images)


def calculate_mean_std(all_images):
	all_means = []
	all_stds = []

	for i, image_path in enumerate(all_images):
		image = imread(image_path, mode='RGB')
		means = np.mean(image, axis=(0, 1))
		all_means.append(means)
		stds = np.std(image, axis=(0, 1))
		all_stds.append(stds)

		sys.stdout.write('\r>> Processed image_path %d/%d' % (i + 1, len(all_images)))
		sys.stdout.flush()

	mean = np.average(all_means, axis=0)
	std = np.average(all_stds, axis=0)

	print('\n')
	print('mean')
	print(mean)
	print('std')
	print(std)
	print()


if __name__ == '__main__':
	main()
