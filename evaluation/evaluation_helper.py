import json
import os
from collections import namedtuple

import matlab.engine
import tensorflow as tf

MatlabEvaluationResult = namedtuple('MatlabEvaluationResult', 'mAP,rank1,rank5,rank10,rank50')
EVALUATION_RESULT_FILE_NAME = 'evaluation.json'


def run_matlab_evaluation(path):
	print('Running Matlab evaluation for path %s' % path)

	eng = matlab.engine.start_matlab()
	eng.cd(r'evaluation/matlab')

	print('Running market evaluation...')
	eval_result = eng.MarketEvalForPath(path)

	ranks = eval_result['rec_rates'][0]
	return MatlabEvaluationResult(mAP=eval_result['mAP'], rank1=ranks[0], rank5=ranks[4], rank10=ranks[9], rank50=ranks[49])


def save_matlab_evaluation(directory, evaluation_result: MatlabEvaluationResult):
	json.dump(evaluation_result._asdict(), open(os.path.join(directory, EVALUATION_RESULT_FILE_NAME), 'w'))


def load_matlab_evaluation(directory):
	file = os.path.join(directory, EVALUATION_RESULT_FILE_NAME)
	if os.path.exists(file):
		return MatlabEvaluationResult(**json.load(open(file, 'r')))
	else:
		return None


class MatlabEvaluationSummaryWriter:
	def __init__(self, output_directory):
		self._evaluation_graph = tf.Graph()

		with self._evaluation_graph.as_default() as graph:
			with tf.name_scope('preid_eval'):
				self._map_placeholder = tf.placeholder(dtype=tf.float32)
				tf.summary.scalar('mAP', self._map_placeholder)
				self._rank1_placeholder = tf.placeholder(dtype=tf.float32)
				tf.summary.scalar('rank1', self._rank1_placeholder)
				self._rank5_placeholder = tf.placeholder(dtype=tf.float32)
				tf.summary.scalar('rank5', self._rank5_placeholder)
				self._rank10_placeholder = tf.placeholder(dtype=tf.float32)
				tf.summary.scalar('rank10', self._rank10_placeholder)
				self._rank50_placeholder = tf.placeholder(dtype=tf.float32)
				tf.summary.scalar('rank50', self._rank50_placeholder)

			self._eval_summary = tf.summary.merge_all()
			self._eval_summary_writer = tf.summary.FileWriter(os.path.join(output_directory, 'eval'), graph)


	def write_evaluation_result(self, global_step, evaluation_result: MatlabEvaluationResult):
		with tf.Session(graph=self._evaluation_graph) as sess:
			summary = sess.run(self._eval_summary, feed_dict={self._map_placeholder: evaluation_result.mAP,
															  self._rank1_placeholder: evaluation_result.rank1,
															  self._rank5_placeholder: evaluation_result.rank5,
															  self._rank10_placeholder: evaluation_result.rank10,
															  self._rank50_placeholder: evaluation_result.rank50})
			self._eval_summary_writer.add_summary(summary, global_step)


def test():
	directory = 'D:/development/private/masters/results/tensorflow-models/market1501/active/2017-09-27_resnet_v1_50_views_v2-pose-maps/predictions-best'
	result = run_matlab_evaluation(directory)
	print(result)
	save_matlab_evaluation(directory, result)


if __name__ == '__main__':
	test()


def get_evaluation_summary_writer(do_evaluation, output_directory):
	if do_evaluation:
		return MatlabEvaluationSummaryWriter(output_directory)
	else:
		return None
