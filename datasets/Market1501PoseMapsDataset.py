import os

import numpy as np

from datasets.Market1501Dataset import Market1501Dataset


class Market1501PoseMapsDataset(Market1501Dataset):
	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=True):
		super().__init__(data_directory=data_directory, dataset_part=dataset_part, mean=mean, std=std, num_classes=num_classes, augment=augment, png=png)
		self._pose_maps_directory = os.path.join(self._data_directory, 'pose-map-files/%s/' % self._dataset_part)

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		def get_pose_map(file_name, _):
			pose_map_path = os.path.join(self._pose_maps_directory, file_name.decode('utf-8') + ".npy")
			loaded = np.load(pose_map_path)
			return loaded

		image_path_tensor, file_name_tensor, actual_label_tensor, label_tensor, camera_tensor = sliced_input_data
		image_tensor = self.read_and_distort_image(file_name_tensor, image_path_tensor, image_size, get_pose_map)
		return self.get_dict_for_batching(actual_label_tensor=actual_label_tensor, camera_tensor=camera_tensor, file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor,
										  image_tensor=image_tensor, label_tensor=label_tensor)
