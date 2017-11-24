from datasets.Cuhk03Dataset import CUHK03_NUM_TRAINING_CLASSES, CUHK03_MEAN, CUHK03_STD, Cuhk03Dataset
from datasets.Market1501PoseMapsDataset import Market1501PoseMapsDataset


class Cuhk03PoseMapsDataset(Market1501PoseMapsDataset):
	def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
		if num_classes is None:
			num_classes = CUHK03_NUM_TRAINING_CLASSES

		super().__init__(mean=CUHK03_MEAN,
		                 std=CUHK03_STD,
		                 num_classes=num_classes,
		                 data_directory=data_directory,
		                 dataset_part=dataset_part,
		                 augment=augment,
		                 png=True)

	@staticmethod
	def get_label_from_path(path):
		return Cuhk03Dataset.get_label_from_path(path)

	@staticmethod
	def get_camera_from_path(path):
		return 0
