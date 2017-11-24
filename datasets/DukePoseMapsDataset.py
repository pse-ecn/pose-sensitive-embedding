from datasets import DukeDataset
from datasets.Market1501PoseMapsDataset import Market1501PoseMapsDataset


class DukePoseMapsDataset(Market1501PoseMapsDataset):
	def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
		if num_classes is None:
			num_classes = DukeDataset.DUKE_NUM_TRAINING_CLASSES

		super().__init__(mean=DukeDataset.DUKE_MEAN, std=DukeDataset.DUKE_STD, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment)
