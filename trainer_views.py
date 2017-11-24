import argparse
import os

import tensorflow as tf
from tensorflow.contrib.learn import RunConfig

from datasets.DatasetFactory import DatasetFactory
from helper.checkpoint_helper import is_initial_run
from helper.model_helper import get_model_function, get_input_function
from nets import nets_factory

slim = tf.contrib.slim


def start_training(data_directory, dataset_name, output_directory, network_name, batch_size, learning_rate, batch_threads, num_epochs, initial_checkpoint, checkpoint_exclude_scopes,
                   ignore_missing_variables, trainable_scopes, fixed_learning_rate, learning_rate_decay_rate, num_classes):
	dataset_factory = DatasetFactory(dataset_name=dataset_name, data_directory=data_directory)
	model_params = {'learning_rate': learning_rate,
	                'fixed_learning_rate': fixed_learning_rate,
	                'learning_rate_decay_rate': learning_rate_decay_rate,
	                'learning_rate_decay_steps': dataset_factory.get_dataset('train').get_number_of_samples() // batch_size}

	run_config = RunConfig(keep_checkpoint_max=10, save_checkpoints_steps=None)
	# Instantiate Estimator
	estimator = tf.estimator.Estimator(
		model_fn=get_model_function(output_directory, network_name, dataset_factory.get_dataset('train').num_classes() if num_classes is None else num_classes, initial_checkpoint,
		                            checkpoint_exclude_scopes, ignore_missing_variables, trainable_scopes),
		params=model_params,
		model_dir=output_directory,
		config=run_config)
	image_size = nets_factory.get_input_size(network_name)

	if not is_initial_run(output_directory):
		run_validation(dataset_factory, batch_size, batch_threads, estimator, image_size)  # do a validation ahead to see the improvement in the first epoch

	for epoch in range(num_epochs):
		run_training(dataset_factory, batch_size, batch_threads, epoch, estimator, num_epochs, image_size)
		run_validation(dataset_factory, batch_size, batch_threads, estimator, image_size)

	run_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size)


def run_evaluation(batch_size, batch_threads, dataset_factory, estimator, image_size):
	print('\n\nRunning test set evaluation:')
	evaluation_result = estimator.evaluate(input_fn=get_input_function(dataset_factory.get_dataset('test'), batch_size, batch_threads, False, image_size))
	print('Finished test evaluation: ')
	print(evaluation_result)


def run_training(dataset_factory, batch_size, batch_threads, epoch, estimator, num_epochs, image_size):
	print('\n\nRunning training of epoch %d of %d:\n' % (epoch + 1, num_epochs))
	train_input_function = get_input_function(dataset_factory.get_dataset('train'), batch_size, batch_threads, True, image_size)
	estimator.train(input_fn=train_input_function)
	print('\nFinished Training epoch %d\n\nRunning Validation' % (epoch + 1))


def run_validation(dataset_factory, batch_size, batch_threads, estimator, image_size):
	val_input_function = get_input_function(dataset_factory.get_dataset('val'), batch_size, batch_threads, False, image_size)
	evaluation_result = estimator.evaluate(input_fn=val_input_function)
	print('Finished Validation: ')
	print(evaluation_result)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', help='Directory to write the output', dest='output_directory')
	parser.add_argument('--data', help='Specify the folder with the images to be trained and evaluated', dest='data_directory')
	parser.add_argument('--dataset-name', help='The name of the dataset')
	parser.add_argument('--batch-size', help='The batch size', type=int, default=16)
	parser.add_argument('--learning-rate', help='The learning rate', type=float, default=0.0001)
	parser.add_argument('--batch-threads', help='The number of threads to be used for batching', type=int, default=8)
	parser.add_argument('--num-epochs', help='The number of epochs to be trained', type=int, default=50)
	parser.add_argument('--initial-checkpoint', help='The initial model to be loaded')
	parser.add_argument('--checkpoint-exclude-scopes', help='Scopes to be excluded when loading initial checkpoint')
	parser.add_argument('--trainable-scopes', help='Scopes which will be trained')
	parser.add_argument('--network-name', help='Name of the network')
	parser.add_argument('--ignore-missing-variables', help='If missing variables should be ignored', action='store_true')
	parser.add_argument('--fixed-learning-rate', help='If set, no exponential learning rate decay is used', action='store_true')
	parser.add_argument('--learning-rate-decay-rate', help='The base of the learning rate decay factor', type=float, default=0.96)
	parser.add_argument('--num-classes', help='Number of classes', type=int, default=None)
	args = parser.parse_args()

	print('Running with command line arguments:')
	print(args)
	print('\n\n')

	# tf.logging.set_verbosity(tf.logging.INFO)

	if not os.path.exists(args.output_directory):
		os.makedirs(args.output_directory)

	start_training(args.data_directory, args.dataset_name, args.output_directory, args.network_name, args.batch_size, args.learning_rate, args.batch_threads, args.num_epochs,
	               args.initial_checkpoint, args.checkpoint_exclude_scopes, args.ignore_missing_variables, args.trainable_scopes, args.fixed_learning_rate, args.learning_rate_decay_rate,
	               args.num_classes)

	print('Exiting ...')


if __name__ == '__main__':
	main()
