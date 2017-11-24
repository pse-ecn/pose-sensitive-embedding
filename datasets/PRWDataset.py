import os
import random

from datasets.Market1501Dataset import Market1501Dataset

PRW_MEAN = [107.78022505, 102.29057153, 100.65454852]
PRW_STD = [48.71011114, 47.32708588, 46.7601729]
PRW_NUM_TRAINING_CLASSES = 485


class PRWDataset(Market1501Dataset):
	def __init__(self, data_directory, dataset_part, mean=None, std=None, num_classes=None, augment=True, png=False):
		if mean is None:
			mean = PRW_MEAN
		if std is None:
			std = PRW_STD
		if num_classes is None:
			num_classes = PRW_NUM_TRAINING_CLASSES

		super().__init__(mean=mean,
		                 std=std,
		                 num_classes=num_classes,
		                 data_directory=data_directory,
		                 dataset_part=dataset_part,
		                 augment=augment,
		                 png=png)

	def get_input_data(self, is_training):
		if self._dataset_part is 'test':
			image_paths = self.get_images_from_folder()

			if is_training:
				random.shuffle(image_paths)

			file_names = [os.path.basename(file) for file in image_paths]

			actual_labels = [0] * len(image_paths)
			labels = [0] * len(image_paths)
			cameras = [0] * len(image_paths)

			print('Read %d image paths for processing for dataset_part: %s' % (len(image_paths), self._dataset_part))
			return image_paths, file_names, actual_labels, labels, cameras

		else:
			return super().get_input_data(is_training)
