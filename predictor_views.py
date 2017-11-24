import argparse
import os
import shutil
import sys

import tensorflow as tf
from tensorflow.contrib.learn import RunConfig

from datasets.DatasetFactory import DatasetFactory
from helper.model_helper import get_model_function, get_input_function
from nets import nets_factory

slim = tf.contrib.slim


def start_prediction(output_directory, data_directory, dataset_name, model_dir, network_name, batch_size, batch_threads, checkpoint_exclude_scopes):
	dataset_factory = DatasetFactory(dataset_name=dataset_name, data_directory=data_directory, augment=False)

	run_config = RunConfig(keep_checkpoint_max=10, save_checkpoints_steps=None)
	# Instantiate Estimator
	estimator = tf.estimator.Estimator(
		model_fn=get_model_function(model_dir, network_name, dataset_factory.get_dataset('train').num_classes(), checkpoint_exclude_scopes=checkpoint_exclude_scopes),
		model_dir=model_dir,
		config=run_config,
		params={})
	image_size = nets_factory.get_input_size(network_name)

	run_prediction_and_evaluation(output_directory, batch_size, batch_threads, dataset_factory, estimator, image_size)


def run_prediction_and_evaluation(output_directory, batch_size, batch_threads, dataset_factory, estimator, image_size):
	predict_views(batch_size, batch_threads, dataset_factory, estimator, image_size, output_directory, 'train')
	predict_views(batch_size, batch_threads, dataset_factory, estimator, image_size, output_directory, 'test')


def predict_views(batch_size, batch_threads, dataset_factory, estimator, image_size, output_directory, dataset_part):
	print('Starting views evaluation...')

	dataset = dataset_factory.get_dataset(dataset_part)

	output_directory = os.path.join(output_directory, dataset_part)
	if os.path.exists(output_directory):
		shutil.rmtree(output_directory)
	os.makedirs(output_directory)

	print('\n\nRunning Prediction for %s' % dataset_part)
	input_function = get_input_function(dataset, batch_size, batch_threads, False, image_size)
	predicted = estimator.predict(input_fn=input_function)
	num_samples = dataset.get_number_of_samples()

	for sample, prediction in enumerate(predicted):
		target_path = os.path.join(output_directory, str(prediction['views_classifications']), prediction['file_names'].decode('UTF-8'))
		original_path = prediction['paths'].decode('UTF-8')

		directory = os.path.dirname(target_path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		shutil.copy(original_path, target_path)

		# print(prediction['views_softmax'])

		if (sample + 1) % batch_size == 0:
			sys.stdout.write('\r>> Processed %d samples of %d' % (sample + 1, num_samples))
		sys.stdout.flush()

	print('Finished views prediction.')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', help='Specify the folder with the images to be trained and evaluated', dest='data_directory')
	parser.add_argument('--dataset-name', help='The name of the dataset')
	parser.add_argument('--batch-size', help='The batch size', type=int, default=16)
	parser.add_argument('--batch-threads', help='The number of threads to be used for batching', type=int, default=4)
	parser.add_argument('--model-dir', help='The model to be loaded')
	parser.add_argument('--network-name', help='Name of the network')
	parser.add_argument('--output', help='Output directory')
	parser.add_argument('--checkpoint-exclude-scopes', help='Scopes to be excluded from the checkpoint')

	args = parser.parse_args()

	print('Running with command line arguments:')
	print(args)
	print('\n\n')

	# tf.logging.set_verbosity(tf.logging.INFO)

	start_prediction(args.output, args.data_directory, args.dataset_name, args.model_dir, args.network_name, args.batch_size, args.batch_threads, args.checkpoint_exclude_scopes)

	print('Exiting ...')


if __name__ == '__main__':
	main()
