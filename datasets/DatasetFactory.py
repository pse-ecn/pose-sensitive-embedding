from datasets.Cuhk03Dataset import Cuhk03Dataset
from datasets.Cuhk03PoseMapsDataset import Cuhk03PoseMapsDataset
from datasets.DukeDataset import DukeDataset
from datasets.DukePoseMapsDataset import DukePoseMapsDataset
from datasets.Market1501Dataset import Market1501Dataset
from datasets.Market1501PoseMapsDataset import Market1501PoseMapsDataset
from datasets.MarsDataset import MarsDataset
from datasets.MarsPoseMapsDataset import MarsPoseMapsDataset
from datasets.PRWDataset import PRWDataset
from datasets.PRWPoseMapsDataset import PRWPoseMapsDataset
from datasets.RapDataset import RapDataset
from datasets.RapPoseMapsDataset import RapPoseMapsDataset


class DatasetFactory:
	def __init__(self, dataset_name, data_directory, augment=True, num_classes=None):
		self._data_directory = data_directory
		self._dataset_name = dataset_name
		self._augment = augment
		self._num_classes = num_classes

	def get_dataset(self, dataset_part):
		if self._dataset_name == 'rap':
			return RapDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		if self._dataset_name == 'rap-pose-maps':
			return RapPoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'market1501':
			return Market1501Dataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'market1501-pose-maps':
			return Market1501PoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'duke':
			return DukeDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'duke-pose-maps':
			return DukePoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'mars':
			return MarsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'mars-pose-maps':
			return MarsPoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'prw':
			return PRWDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'prw-pose-maps':
			return PRWPoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'cuhk03':
			return Cuhk03Dataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		elif self._dataset_name == 'cuhk03-pose-maps':
			return Cuhk03PoseMapsDataset(data_directory=self._data_directory, dataset_part=dataset_part, augment=self._augment, num_classes=self._num_classes)

		else:
			raise ValueError('Unknown dataset name: %s' % self._data_directory)

	def get_dataset_name(self):
		return self._dataset_name
