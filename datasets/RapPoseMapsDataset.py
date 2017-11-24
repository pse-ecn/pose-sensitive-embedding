import os

import numpy as np

from datasets.RapDataset import RapDataset


class RapPoseMapsDataset(RapDataset):
	def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
		super().__init__(data_directory=data_directory, dataset_part=dataset_part, augment=augment, num_classes=num_classes)
		self._pose_maps = {}

	def get_input_data(self, is_training):
		print('Reading pose maps...')
		self._pose_maps = self.load_pose_maps_file()
		print('Finished reading pose maps.')

		return super().get_input_data(is_training)

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		def get_pose_map(file_name, path):
			return self._pose_maps[file_name.decode('utf-8')[7:]]

		file_name_tensor, image_path_tensor, label_tensor, view_tensor = sliced_input_data
		image_tensor = self.read_and_distort_image(file_name_tensor, image_path_tensor, image_size, get_pose_map)
		return self.get_dict_for_batching(file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor, multi_class_label=label_tensor, image_tensor=image_tensor, view_label=view_tensor)

	def load_pose_maps_file(self):
		pose_maps_path = os.path.join(self._data_directory, 'pose_maps/poses_maps.npy')
		pose_maps = np.load(pose_maps_path).item()
		return pose_maps
