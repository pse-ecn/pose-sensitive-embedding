from datasets.Market1501Dataset import Market1501Dataset

DUKE_MEAN = [112.18328802, 109.87745459, 113.67777469]
DUKE_STD = [50.94881722, 51.72869758, 48.58530332]
DUKE_NUM_TRAINING_CLASSES = 702


class DukeDataset(Market1501Dataset):
	def __init__(self, data_directory, dataset_part, augment=True, num_classes=None):
		if num_classes is None:
			num_classes = DUKE_NUM_TRAINING_CLASSES

		super().__init__(mean=DUKE_MEAN, std=DUKE_STD, num_classes=num_classes, data_directory=data_directory, dataset_part=dataset_part, augment=augment)
