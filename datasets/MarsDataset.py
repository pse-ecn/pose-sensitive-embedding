import glob
import ntpath
import os

from datasets.Market1501Dataset import Market1501Dataset


class MarsDataset(Market1501Dataset):
	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=False):
		if mean is None:
			mean = [105.67011756, 100.26972989, 97.50862173]
		if std is None:
			std = [49.19647396, 47.93514119, 47.16492013]
		if num_classes is None:
			num_classes = 625

		super().__init__(mean=mean,
		                 std=std,
		                 num_classes=num_classes,
		                 data_directory=data_directory,
		                 dataset_part=dataset_part,
		                 augment=augment,
		                 png=png)

	def get_images_from_folder(self):
		if self._dataset_part is 'train':
			return glob.glob(os.path.join(self._data_directory, 'bbox_train', '*', '*.jpg'))

		elif self._dataset_part is 'test':
			data_file = os.path.join(self._data_directory, 'test_name.txt')

			with open(data_file) as file:
				all_image_names = file.read().splitlines()

			all_images = [os.path.join(self._data_directory, 'bbox_test', image_name[:4], image_name) for image_name in all_image_names]

			return all_images

		else:
			raise ValueError('unknown dataset part')

	@staticmethod
	def get_label_from_path(path):
		filename = ntpath.basename(path)
		label = filename[:4]
		return int(label[:3].lstrip('0') + label[3])  # so complicated to make 00-1 and 0000 work at the same time

	@staticmethod
	def get_camera_from_path(path):
		filename = ntpath.basename(path)
		camera_sequence_string = filename[5]
		return int(camera_sequence_string)
