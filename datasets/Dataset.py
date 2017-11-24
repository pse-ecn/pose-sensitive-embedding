from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

CROP_BORDER_PERCENT = 0.05


class Dataset(ABC):
	def __init__(self, mean, std, num_classes, data_directory, dataset_part, augment=True, png=True):
		self._mean = mean
		self._std = std
		self._num_classes = num_classes
		self._data_directory = data_directory
		self._dataset_part = dataset_part
		self._augment = augment
		self._png = png

	def mean(self):
		return self._mean

	def std(self):
		return self._std

	def num_classes(self):
		return self._num_classes

	@abstractmethod
	def get_input_data(self, is_training):
		pass

	@abstractmethod
	def get_number_of_samples(self):
		pass

	@abstractmethod
	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		pass

	@abstractmethod
	def get_input_function_dictionaries(self, batched_input_data):
		pass

	def dataset_part(self):
		return self._dataset_part

	def _read_and_normalize_image_quadratic(self, path_tensor, image_size):
		return self._read_and_normalize_image(path_tensor, height=image_size, width=image_size)

	def _read_and_normalize_image(self, path_tensor, height, width):
		if self._png:
			image_tensor = tf.image.decode_png(tf.read_file(path_tensor), channels=3)
		else:
			image_tensor = tf.image.decode_jpeg(tf.read_file(path_tensor), channels=3)

		resized_image_tensor = tf.image.resize_images(image_tensor, size=[height, width])
		mean = np.reshape(self.mean(), [1, 1, 3])
		std = np.reshape(self.std(), [1, 1, 3])
		normalized_image_tensor = (resized_image_tensor - mean) / std
		return normalized_image_tensor

	def read_and_distort_image(self, file_name_tensor, image_path_tensor, image_size, get_pose_map_fn=None):
		if get_pose_map_fn is not None:
			pose_map = tf.py_func(get_pose_map_fn, [file_name_tensor, image_path_tensor], tf.float32, stateful=False)
			pose_map.set_shape([None, None, 14])
		else:
			pose_map = None

		if self._augment and self._dataset_part == 'train':
			height = int(image_size * (1 + CROP_BORDER_PERCENT))
			width = int(image_size * (1 + 2 * CROP_BORDER_PERCENT))
			image_tensor = self._read_and_normalize_image(image_path_tensor, height, width)

			with tf.name_scope('original'):
				self.add_image_summary(image_tensor)

			if pose_map is not None:
				resized_pose_map = tf.image.resize_images(pose_map, size=[height, width])
				image_tensor = tf.concat([image_tensor, resized_pose_map], axis=-1)

			with tf.name_scope('concatenated'):
				self.add_image_summary(image_tensor)

			with tf.name_scope('random-cropped'):
				image_tensor = tf.random_crop(image_tensor, [image_size, image_size, image_tensor.get_shape().as_list()[2]])
				self.add_image_summary(image_tensor)

			with tf.name_scope('random-flipped'):
				image_tensor = tf.image.random_flip_left_right(image_tensor)
				self.add_image_summary(image_tensor)


		else:
			image_tensor = self._read_and_normalize_image(image_path_tensor, image_size, image_size)

			if pose_map is not None:
				resized_pose_map = tf.image.resize_images(pose_map, size=[image_size, image_size])
				image_tensor = tf.concat([image_tensor, resized_pose_map], axis=-1)

		return image_tensor

	@staticmethod
	def add_image_summary(image_tensor):
		# tf.summary.image('image', tf.expand_dims(image_tensor[:, :, 0:3], axis=0))
		# Dataset.add_pose_map_summary(image_tensor)
		pass

	@staticmethod
	def add_pose_map_summary(image):
		if image.shape[2] > 3:
			tf.summary.image('pose_maps', [tf.reduce_max(image[:, :, 4:], axis=-1, keep_dims=True)])

	@staticmethod
	def get_dict_for_batching(actual_label_tensor=None, camera_tensor=None, file_name_tensor=None, image_path_tensor=None, image_tensor=None, label_tensor=None, view_label=None,
	                          multi_class_label=None):
		dictionary = {}

		if actual_label_tensor is not None:
			dictionary['actual_label'] = actual_label_tensor

		if camera_tensor is not None:
			dictionary['camera'] = camera_tensor

		if file_name_tensor is not None:
			dictionary['file_name'] = file_name_tensor

		if image_path_tensor is not None:
			dictionary['path'] = image_path_tensor

		if image_tensor is not None:
			dictionary['image'] = image_tensor

		if label_tensor is not None:
			dictionary['label'] = label_tensor

		if view_label is not None:
			dictionary['view'] = view_label

		if multi_class_label is not None:
			dictionary['multi_class_label'] = multi_class_label

		return dictionary
