import os

from datasets.Dataset import Dataset

RAP_MEAN = [105.98118096, 105.37399591, 100.98540261]
RAP_STD = [58.20315763, 53.74679653, 55.90117479]


class RapDataset(Dataset):
	FILE_BY_PART = {'train': 'train_list_52.txt', 'test': 'test_list_52.txt', 'val': 'val_list_52.txt'}
	CROP_BORDER = 0.05

	def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
		if num_classes is None:
			num_classes = 51

		super().__init__(mean=RAP_MEAN, std=RAP_STD, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment)

	def get_input_data(self, is_training):
		data_file_name = self.get_data_file_for_mode()

		file_name_list = []
		paths_list = []
		labels_list = []
		views_list = []

		with open(data_file_name, 'r') as reader:
			for line in reader.readlines():
				space_split = line.split(' ')

				file_name = space_split[0]
				labels_with_view = list(map(int, list(space_split[1].split(','))[:-1]))

				labels = labels_with_view[:-1]
				view = labels_with_view[-1]

				file_name_list.append(file_name)
				paths_list.append(os.path.join(self._data_directory, file_name))
				labels_list.append(labels)
				views_list.append(view)

		print('Read %d image paths for processing from %s' % (len(file_name_list), data_file_name))

		return file_name_list, paths_list, labels_list, views_list

	def get_number_of_samples(self):
		data_file = self.get_data_file_for_mode()

		with open(data_file, 'r') as reader:
			return len(reader.readlines())

	def prepare_sliced_data_for_batching(self, sliced_input_data, image_size):
		file_name_tensor, image_path_tensor, label_tensor, view_tensor = sliced_input_data
		image_tensor = self.read_and_distort_image(file_name_tensor, image_path_tensor, image_size)
		return self.get_dict_for_batching(file_name_tensor=file_name_tensor, image_path_tensor=image_path_tensor, multi_class_label=label_tensor, image_tensor=image_tensor, view_label=view_tensor)

	def get_input_function_dictionaries(self, batched_input_data):
		return {'file_names': batched_input_data['file_name'], 'paths': batched_input_data['path'], 'images': batched_input_data['image']}, \
			   {'multi_class_labels': batched_input_data['multi_class_label'], 'views': batched_input_data['view']}

	def get_data_file_for_mode(self):
		data_file = self.FILE_BY_PART[self._dataset_part]
		return os.path.join(self._data_directory, data_file)
