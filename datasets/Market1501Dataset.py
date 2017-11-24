import glob
import ntpath
import os
import random

from datasets.Dataset import Dataset


class Market1501Dataset(Dataset):
	FILE_BY_PART = {'train': 'bounding_box_train', 'test': 'bounding_box_test', 'query': 'query', 'distractors': 'distractors500k'}

	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=True):
		if mean is None:
			mean = [105.69332121, 99.12930469, 97.90910844]
		if std is None:
			std = [50.26135204, 48.62204008, 48.24029389]
		if num_classes is None:
			num_classes = 751

		super().__init__(mean=mean, std=std, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment, png=png)

	def get_input_data(self, is_training):
		image_paths = self.get_images_from_folder()

		if is_training:
			random.shuffle(image_paths)

		file_names = [os.path.basename(file) for file in image_paths]

		actual_labels = [self.get_label_from_path(image_path) for image_path in image_paths]
		label_mapping = {label: index for index, label in enumerate(list(sorted(set(actual_labels))))}
		labels = [label_mapping[actual_label] for actual_label in actual_labels]
		cameras = [self.get_camera_from_path(image_path) for image_path in image_paths]

		print('Read %d image paths for processing for dataset_part: %s' % (len(image_paths), self._dataset_part))
		return image_paths, file_names, actual_labels, labels, cameras

	def get_number_of_samples(self):
		return len(self.get_images_from_folder())

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		image_path_tensor, file_name_tensor, actual_label_tensor, label_tensor, camera_tensor = sliced_input_data
		image_tensor = self.read_and_distort_image(file_name_tensor, image_path_tensor, image_size)

		return self.get_dict_for_batching(actual_label_tensor=actual_label_tensor, camera_tensor=camera_tensor, file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor,
										  image_tensor=image_tensor, label_tensor=label_tensor)

	def get_images_from_folder(self):
		data_file = self.get_data_file()
		return self.get_png_and_jpg(data_file)

	@staticmethod
	def get_png_and_jpg(data_file):
		all_images = glob.glob(os.path.join(data_file, '*.png'))
		all_images.extend(glob.glob(os.path.join(data_file, '*.jpg')))
		return all_images

	def get_data_file(self):
		data_file = self.FILE_BY_PART[self._dataset_part]
		return os.path.join(self._data_directory, data_file)

	def get_input_function_dictionaries(self, batched_input_data):
		return {'paths': batched_input_data['path'], 'images': batched_input_data['image'], 'file_names': batched_input_data['file_name']}, \
			   {'labels': batched_input_data['label'], 'actual_labels': batched_input_data['actual_label'], 'cameras': batched_input_data['camera']}

	@staticmethod
	def get_label_from_path(path):
		filename = ntpath.basename(path)
		label = filename.split('_')[0]
		return int(label)

	@staticmethod
	def get_camera_from_path(path):
		filename = ntpath.basename(path)
		camera_sequence_string = filename.split('_')[1]
		return int(camera_sequence_string[1:2])
