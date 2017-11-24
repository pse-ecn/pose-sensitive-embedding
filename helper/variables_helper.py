import tensorflow as tf


def get_variable_names(variables):
	return [get_variable_name(variable) for variable in variables]


def get_variables_excluding(variables, exclude_scopes):
	if exclude_scopes is None:
		return variables

	cleaned_variables = []
	for variable in variables:
		if not is_variable_excluded(variable, exclude_scopes):
			cleaned_variables.append(variable)

	return cleaned_variables


def is_variable_excluded(variable, exclude_scopes):
	variable_name = get_variable_name(variable)

	for excluded_scope in exclude_scopes.split(','):
		if variable_name.startswith(excluded_scope):
			return True
	return False


def get_variable_name(variable):
	return variable.name.split(':')[0]


def get_variables_in_scopes(collection_key, scopes):
	if scopes is None:
		return tf.get_collection(collection_key)

	all_variables = []

	for scope in scopes.split(','):
		all_variables.extend(tf.get_collection(collection_key, scope))

	return all_variables


def get_training_variables(collection_key, in_scopes=None, exclude_scopes=None):
	variables = get_variables_in_scopes(collection_key, in_scopes)
	variables = get_variables_excluding(variables, exclude_scopes)
	return variables
