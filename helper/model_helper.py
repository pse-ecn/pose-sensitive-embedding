import tensorflow as tf

from helper import variables_helper
from helper.checkpoint_helper import check_init_from_initial_checkpoint
from nets import nets_factory

ModeKeys = tf.estimator.ModeKeys


def get_input_function(dataset, batch_size, batch_threads, is_training, image_size):
	input_data = dataset.get_input_data(is_training)

	def input_fn():
		sliced_input_data = tf.train.slice_input_producer(input_data, num_epochs=1, shuffle=is_training, capacity=4096)
		sliced_data_dictionary = dataset.prepare_sliced_data_for_batching(sliced_input_data, image_size)

		batched_input_data = tf.train.batch(tensors=sliced_data_dictionary,
		                                    batch_size=batch_size,
		                                    num_threads=batch_threads,
		                                    capacity=batch_threads * batch_size * 2,
		                                    allow_smaller_final_batch=not is_training)
		# tf.summary.image(name='input_images', tensor=batched_input_data['image'])

		(features, targets) = dataset.get_input_function_dictionaries(batched_input_data)
		features.update(targets)
		return features, targets

	return input_fn


def get_model_function(output_directory, network_name, num_classes, initial_checkpoint=None, checkpoint_exclude_scopes=None, ignore_missing_variables=False, trainable_scopes=None,
                       not_trainable_scopes=None):
	def model_fn(features, labels, mode, params):
		if labels is None:  # when predicting, labels is None
			labels = {}

		images = features['images']
		file_names = features['file_names']
		labels_tensor = labels['labels'] if 'labels' in labels else None
		mse_labels = labels['mse_labels'] if 'mse_labels' in labels else None

		network_function = nets_factory.get_network_fn(network_name, num_classes, weight_decay=0.00004, is_training=mode == ModeKeys.TRAIN)
		logits, end_points = network_function(images)

		aux_logits = end_points['AuxLogits'] if 'AuxLogits' in end_points else None
		views_labels = labels['views'] if 'views' in labels else None
		views_logits = end_points['PoseLogits'] if 'PoseLogits' in end_points else None

		check_init_from_initial_checkpoint(output_directory, initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables)

		predictions_dict = {}
		train_op = tf.no_op()
		eval_metric_ops = {}

		if mode == ModeKeys.EVAL or mode == ModeKeys.TRAIN:
			with tf.name_scope('losses'):
				tf.summary.scalar(name='regularization', tensor=tf.losses.get_regularization_loss())

				with tf.name_scope('softmax_cross_entropy'):
					if labels_tensor is not None:
						tf.summary.scalar(name='logits', tensor=tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor, logits=logits, scope='logits'))
						tf.summary.scalar(name='training-top-1', tensor=tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=labels_tensor, k=1), tf.float32)))
						tf.summary.scalar(name='training-top-5', tensor=tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=labels_tensor, k=5), tf.float32)))

						if aux_logits is not None:
							tf.summary.scalar(name='auxLogits', tensor=tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor, logits=aux_logits, scope='aux_logits'))
							tf.summary.scalar(name='training-aux-top-1', tensor=tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=aux_logits, targets=labels_tensor, k=1), tf.float32)))
							tf.summary.scalar(name='training-aux-top-5', tensor=tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=aux_logits, targets=labels_tensor, k=5), tf.float32)))

					if views_logits is not None and views_labels is not None:
						tf.summary.scalar(name='3_views', tensor=tf.losses.sparse_softmax_cross_entropy(labels=views_labels, logits=views_logits, scope='3_views'))

				with tf.name_scope('mean_squared_error'):
					if mse_labels is not None:
						tf.summary.scalar(name='logits', tensor=tf.losses.mean_squared_error(labels=mse_labels, predictions=logits, scope='logits'))

						if aux_logits is not None:
							tf.summary.scalar(name='auxLogits', tensor=tf.losses.mean_squared_error(labels=mse_labels, predictions=aux_logits, scope='aux_logits'))

		if mode == ModeKeys.TRAIN:
			def learning_rate_decay_function(learning_rate, global_step):
				if not params['fixed_learning_rate']:
					return tf.train.exponential_decay(learning_rate=learning_rate,
					                                  global_step=global_step,
					                                  decay_steps=params['learning_rate_decay_steps'],
					                                  decay_rate=params['learning_rate_decay_rate'],
					                                  staircase=True,
					                                  name='learning-rate-decay')
				else:
					return learning_rate

			variables_to_train = variables_helper.get_training_variables(tf.GraphKeys.TRAINABLE_VARIABLES, trainable_scopes, not_trainable_scopes)

			train_op = tf.contrib.layers.optimize_loss(loss=tf.losses.get_total_loss(),
			                                           global_step=tf.train.get_or_create_global_step(),
			                                           learning_rate=params['learning_rate'],
			                                           optimizer=lambda learning_rate: tf.train.AdamOptimizer(learning_rate),
			                                           variables=variables_to_train,
			                                           learning_rate_decay_fn=learning_rate_decay_function)

		if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL:
			predictions_dict = {'logits': logits,
			                    'classifications': tf.argmax(logits, axis=-1),
			                    'file_names': file_names}
			predictions_dict.update(features)
			predictions_dict.update(labels)

			if aux_logits is not None:
				predictions_dict['aux_classifications'] = tf.argmax(aux_logits, axis=-1)

			if views_logits is not None:
				predictions_dict['views_classifications'] = tf.argmax(views_logits, axis=-1)
				predictions_dict['views_softmax'] = tf.nn.softmax(views_logits)

			if 'PreLogits' in end_points:
				predictions_dict['pre_logits'] = end_points['PreLogits']

		if mode == ModeKeys.EVAL:
			if labels_tensor is not None:
				eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels_tensor, predictions_dict['classifications'])}

				if aux_logits is not None:
					eval_metric_ops['aux_accuracy'] = tf.metrics.accuracy(labels_tensor, predictions_dict['aux_classifications'])

			if views_logits is not None and views_labels is not None:
				eval_metric_ops['views_accuracy'] = tf.metrics.accuracy(views_labels, predictions_dict['views_classifications'])

		total_loss = tf.losses.get_total_loss() if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL else None
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, loss=total_loss, train_op=train_op,
		                                  eval_metric_ops=eval_metric_ops)

	return model_fn
