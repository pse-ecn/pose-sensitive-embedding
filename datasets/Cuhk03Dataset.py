import ntpath

from datasets.Market1501Dataset import Market1501Dataset

CUHK03_MEAN = [92.53974645, 91.5915331, 89.44749673]
CUHK03_STD = [59.24914549, 60.11329685, 60.34548437]
CUHK03_NUM_TRAINING_CLASSES = 767


class Cuhk03Dataset(Market1501Dataset):
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
		filename = ntpath.basename(path)
		split_filename = filename.split('_')
		label = split_filename[0] + split_filename[1]
		return int(label)

	@staticmethod
	def get_camera_from_path(path):
		return 0
