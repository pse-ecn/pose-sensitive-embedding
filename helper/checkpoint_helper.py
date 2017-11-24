import os

import tensorflow as tf

from helper import variables_helper
from helper.variables_helper import get_variable_name

STANDARD_EXCLUDE_SCOPES = 'global_step'


def check_init_from_initial_checkpoint(output_directory, initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables):
	if initial_checkpoint is not None and is_initial_run(output_directory):
		print('Initial run: ')
		init_from_checkpoint(initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables)


def init_from_checkpoint(initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables):
	print('Loading initial_checkpoint: %s' % initial_checkpoint)
	variables_dictionary = get_variables_to_restore(initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables)
	tf.contrib.framework.init_from_checkpoint(initial_checkpoint, variables_dictionary)
	print('Finished loading initial_checkpoint')


def get_variables_to_restore(initial_checkpoint, checkpoint_exclude_scopes, ignore_missing_variables):
	checkpoint_exclude_scopes = checkpoint_exclude_scopes + ',' + STANDARD_EXCLUDE_SCOPES if checkpoint_exclude_scopes is not None else STANDARD_EXCLUDE_SCOPES

	variables_list = variables_helper.get_variables_excluding(tf.global_variables(), checkpoint_exclude_scopes)

	if ignore_missing_variables:
		checkpoint_variables = tf.contrib.framework.list_variables(initial_checkpoint)
		checkpoint_variable_names = [variable_name for (variable_name, _) in checkpoint_variables]
		variables_list = [variable for variable in variables_list if get_variable_name(variable) in checkpoint_variable_names]

	return {get_variable_name(variable): variable for variable in variables_list}


def is_initial_run(output_directory):
	return not os.path.exists(os.path.join(output_directory, 'checkpoint'))
