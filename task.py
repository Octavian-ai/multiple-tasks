
"""
Run a network against multiple tasks sequentially to see how task switchover affects learning.

See README.md for how to use this code and a description of what it does.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from datetime import datetime 
import cv2
import os.path
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
IMAGE_SIZE = 28
TEST_N = 10 * 1000

def make_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--input-style', type=str, default='combined')

	parser.add_argument('--fake-data', nargs='?', const=True, type=bool,
											default=False,
											help='If true, uses fake data for unit testing.')
	parser.add_argument('--max-steps', type=int, default=25000,
											help='Number of steps to run trainer.')
	parser.add_argument('--learning-rate', type=float, default=0.01,
											help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
											help='Keep probability for training dropout.')
	parser.add_argument('--optimizer', type=str, default="sgd")
	parser.add_argument('--test-frequency', type=int, default=100)
	parser.add_argument('--train-metadata-frequency', type=int, default=10000)
	parser.add_argument('--trace', action='store_true')
	parser.add_argument('--train-summary-frequency', type=int, default=10)
	parser.add_argument('--batch', type=int, default=100)
	parser.add_argument('--network', nargs='+', default=['conv', 'fc'])
	parser.add_argument('--use-pickle', action='store_true')

	parser.add_argument('--task-time', type=int, default=2000)
	# parser.add_argument('--task-n', type=int, default=1)
	parser.add_argument('--tasks', nargs='+', default=['id'])

	parser.add_argument(
			'--data-dir',
			type=str,
			default='/tmp/tensorflow/mnist/input_data',
			help='Directory for storing input data')
	
	parser.add_argument(
			'--data-dog-dir',
			type=str,
			default='/Users/davidmack/dev/data/dog_breed',
			help='Directory for storing input data')
	parser.add_argument('--pickle-file', type=str, default='images.p')

	parser.add_argument(
			'--log-dir',
			type=str,
			default='./log/test',
			help='Summaries log directory')

	
	return parser


def load_dogs():

	df = pd.read_csv(FLAGS.data_dog_dir + '/' + 'labels.csv')
	df_test = pd.read_csv(FLAGS.data_dog_dir + '/sample_submission.csv')
	
	breed = set(list(set(df['breed']))[0:10])
	n_class = len(breed)

	class_to_num = dict(zip(breed, range(n_class)))
	num_to_class = dict(zip(range(n_class), breed))

	df = df[ df['breed'].isin(breed) ]
	df.head()
	n = len(df)

	# Stop re-doing this
	if os.path.isfile(FLAGS.data_dog_dir + '/' + FLAGS.pickle_file) and FLAGS.use_pickle:
		(x, y) = pickle.load(open(FLAGS.data_dog_dir + '/' + FLAGS.pickle_file, "rb"))

	else:
		x = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
		y = np.zeros((n, n_class), dtype=np.uint8)
		
		i = 0
		for index, row in df.iterrows():
			x[i] = cv2.resize(cv2.imread(FLAGS.data_dog_dir + '/train/%s.jpg' % row['id']), (IMAGE_SIZE, IMAGE_SIZE))
			y[i][class_to_num[row['breed']]] = 1.0
			i += 1

		x = (np.array(x, np.float32) / 255.)
		
		if FLAGS.use_pickle:
			pickle.dump((x, y), open(FLAGS.data_dog_dir + '/' + FLAGS.pickle_file, "wb" ))

	train_percent = 0.9
	train_n = int(len(x) * train_percent)
	test_n = int(len(x) * (1-train_percent))
	reps = (TEST_N // test_n) + 1

	test_image = np.repeat(x[-test_n:], reps, axis=0)[0:TEST_N]
	test_label = np.repeat(y[-test_n:], reps, axis=0)[0:TEST_N]

	return x[0:train_n], y[0:train_n], test_image, test_label 
			

def build_tasks():

	dog_train_image, dog_train_label, dog_test_image, dog_test_label = load_dogs()

	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir,
		one_hot=True,
		fake_data=FLAGS.fake_data)

	def identity(x, y): 
		return x, y

	def reflect(x, y):
		return [i[::-1] for i in x], y

	def invert(x, y):
		return [[1.0-cell for cell in row] for row in x], y

	def noise(x, y):
		return [[np.random.random_sample() for cell in row] for row in x], y

	def zero(x, y): 
		return [[0 for cell in row] for row in x], y

	def one(x, y): 
		return [[1 for cell in row] for row in x], y

	label_perm = np.random.permutation(10)
	label_perm_matrix = [
		[
			(
				1.0 if label_perm[i] == j else 0.0
			) for j in range(10)
		] for i in range(10)
	]
	
	def permutation(x, y):
		return x, [np.matmul(label_perm_matrix, label) for label in y]


	def combine_tasks(task, task_data):
		zero = np.zeros(task_data.shape[:3] + (1,)) # One channel worth of zeros

		if FLAGS.input_style == 'combined':
			reformat = {
				"mnist": lambda x: np.concatenate((x,zero,zero), axis=-1),
				"dog": lambda x: x
			}
		elif FLAGS.input_style == 'channelplex':
			reformat = {
				"mnist": lambda x: np.concatenate((x,    zero, zero, zero), axis=-1),
				"dog":   lambda x: np.concatenate((zero, x), axis=-1)
			}

		return reformat[task](task_data)


	# stupid scoping hack
	dog_batch = [0]

	def dog(test):
		def make_chan(x): 
			return combine_tasks("dog", x)
			
		if test:
			return make_chan(dog_test_image), dog_test_label 
		else:
			x = dog_train_image[dog_batch[0] : dog_batch[0] + FLAGS.batch]
			y = dog_train_label[dog_batch[0] : dog_batch[0] + FLAGS.batch]
			dog_batch[0] = (dog_batch[0] + FLAGS.batch) % (len(dog_train_image) - FLAGS.batch)
			return make_chan(x), y
	

	def with_mnist(fn):
		def make_chan(x): 
			reshaped = np.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
			return combine_tasks("mnist", reshaped)

		def input_fn(test):
			if test:
				return fn(make_chan(mnist.test.images), mnist.test.labels)
			else:
				x, y = mnist.train.next_batch(FLAGS.batch, fake_data=FLAGS.fake_data)
				return fn(make_chan(x), y)

		return input_fn


	return {
		"id": 		with_mnist(identity),
		"ref": 		with_mnist(reflect),
		"inv": 		with_mnist(invert),
		"noise": 	with_mnist(noise),
		"perm": 	with_mnist(permutation),
		"zero": 	with_mnist(zero),
		"one": 		with_mnist(one),
		"dog": 		dog
	}


def train(optimizer):

	task_dict = build_tasks()

	tasks = [ task_dict[i] for i in FLAGS.tasks ]

	task_dir = "t_" + "_".join(FLAGS.tasks)
	net_dir = "n_" + "_".join(FLAGS.network)

	run_tag = task_dir + '/' + FLAGS.input_style + '/' + optimizer.get_name() + '/' + net_dir + '/tt' + str(FLAGS.task_time) + '/' + str(datetime.now())

	sess = tf.InteractiveSession()

	keep_prob = tf.placeholder(tf.float32)

	device_for_graph = '/gpu:0' if FLAGS.gpu else '/cpu:0'
	device_for_measure = '/cpu:0'

	if FLAGS.input_style == 'channelplex':
		CHANNELS = 4
	elif FLAGS.input_style == 'combined':
		CHANNELS = 3

	# We can't initialize these variables to 0 - the network will get stuck.
	def weight_variable(shape):
		"""Create x weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		"""Create x bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def variable_summaries(var):
		"""Attach x lot of summaries to x Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			# tf.summary.scalar('mean', mean)
			# with tf.name_scope('stddev'):
			#   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			# tf.summary.scalar('stddev', stddev)
			# tf.summary.scalar('max', tf.reduce_max(var))
			# tf.summary.scalar('min', tf.reduce_min(var))
			# tf.summary.histogram('histogram', var)

	def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Reusable code for making x simple neural net layer.

		It does x matrix multiply, bias add, and then uses ReLU to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds x number of summary ops.
		"""
		# Adding x name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):

			input_tensor_flat = tf.reshape(input_tensor, [-1]+[np.prod(input_dim)])

			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable([np.prod(input_dim), output_dim])
				variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = bias_variable([output_dim])
				variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor_flat, weights) + biases
				# tf.summary.histogram('pre_activations', preactivate)
			activations = act(preactivate, name='activation')

			# tf.summary.histogram('activations', activations)
			return activations

	def conv_layer(input_tensor, input_dim, kernel_dim, layer_name, act=tf.nn.relu):
		"""Reusable code for making x simple neural net layer.

		It does x matrix multiply, bias add, and then uses ReLU to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds x number of summary ops.
		"""
		# Adding x name scope ensures logical grouping of the layers in the graph.

		input_square = tf.reshape(input_tensor, [-1]+input_dim) 

		with tf.name_scope(layer_name):
			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable(kernel_dim)
				variable_summaries(weights)

			with tf.name_scope('biases'):
				biases = bias_variable(kernel_dim[-1:])
				variable_summaries(biases)
			
			conv = tf.nn.conv2d(input_square, weights, [1, 1, 1, 1], padding='SAME')
			pre_activation = tf.nn.bias_add(conv, biases)

			activation = act(pre_activation, name='activation')
			# tf.summary.histogram('activations', activation)

			pool1 = tf.nn.max_pool(activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
								   padding='SAME', name='pool1')
			norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
							  name='norm1')

			with tf.name_scope('dropout'):
				# tf.summary.scalar('dropout_keep_probability', keep_prob)
				dropped = tf.nn.dropout(norm1, keep_prob)

		return dropped

	def split_list(alist, wanted_parts=1):
		length = len(alist)
		return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
						 for i in range(wanted_parts) ]


	def feed_dict(train, step, test_task=None):
		"""Make x TensorFlow feed_dict: maps data onto Tensor placeholders."""
		nxs = []
		nys = []

		if train or FLAGS.fake_data:
			task_i = (step // FLAGS.task_time) % len(tasks)
			task = tasks[task_i]
			nxs, nys = task(test=False)

			k = FLAGS.dropout
		
		else:
			if test_task is None:
				for task in tasks:
					xs, ys = task(test=True)
					nxs.extend(xs)
					nys.extend(ys)

			else:
				xs, ys = tasks[test_task](test=True)
				nxs.extend(xs)
				nys.extend(ys)

			k = 1.0
		return {x: nxs, y_: nys, keep_prob: k}


	# Let's build our graph! --------------------------------------------------

	with tf.device(device_for_graph):

		# Input placeholders
		with tf.name_scope('input'):
			x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name='x-input')
			y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

		# Kernels
		k1 = [5, 5, CHANNELS, 64]
		k2 = [5, 5, k1[-1], 128]

		# Layer sizes
		s1 = [IMAGE_SIZE, IMAGE_SIZE, CHANNELS]
		s2 = [IMAGE_SIZE//2, IMAGE_SIZE//2, k1[-1]]
		s3 = [IMAGE_SIZE//4, IMAGE_SIZE//4, k2[-1]]

		# Keras removes this pain :P
		if FLAGS.network[0] == 'fc':
			hidden1 = nn_layer(x, s1, 500, 'layer1')

			if FLAGS.network[1] == 'fc':
				y = nn_layer(hidden1, [500], 10, 'layer2', act=tf.identity)

		if FLAGS.network[0] == 'conv':
			hidden1 = conv_layer(x, 
				s1, 
				k1,
				'layer1')

			if FLAGS.network[1] == 'fc':
				y = nn_layer(hidden1, s2, 10, 'layer2', act=tf.identity)

		if FLAGS.network[1] == 'conv':
			hidden2 = conv_layer(hidden1, 
				s2, 
				k2,
				'layer2')

			if FLAGS.network[2] == 'fc':
				y = nn_layer(hidden2, s3, 10, 'layer3', act=tf.identity)

		# Do not apply softmax activation yet, see below.

		with tf.name_scope('cross_entropy'):
			# The raw formulation of cross-entropy,
			#
			# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
			#                               reduction_indices=[1]))
			#
			# can be numerically unstable.
			#
			# So here we use tf.nn.softmax_cross_entropy_with_logits on the
			# raw outputs of the nn_layer above, and then average across
			# the batch.
			diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
			with tf.name_scope('total'):
				cross_entropy = tf.reduce_mean(diff)

		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		with tf.name_scope('train'):
			train_step = optimizer.minimize(cross_entropy)


	# Merge all the summaries and write them out to
	
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/'+run_tag + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/'+run_tag + '/test')
	task_writer = [
		tf.summary.FileWriter(FLAGS.log_dir + '/'+run_tag + '/test_task_'+str(i)+"_"+t) for i, t in enumerate(FLAGS.tasks)
	]

	tf.global_variables_initializer().run()


	with tf.device(device_for_measure):

		tf.summary.scalar('accuracy', accuracy)
		merged = tf.summary.merge_all()

		for step in range(FLAGS.max_steps):
			# Time to test
			if step % FLAGS.test_frequency == 0: 
				summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, step))
				test_writer.add_summary(summary, step)
				print('Accuracy at step %s: %s' % (step, acc))

				for task_i, task in enumerate(tasks):
					summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, step, task_i))
					task_writer[task_i].add_summary(summary, step)

			# Time to train
			else:
				if step % FLAGS.train_metadata_frequency == 1:  # Record execution stats
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					summary, _ = sess.run([merged, train_step],
																feed_dict=feed_dict(True, step),
																options=run_options,
																run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
					train_writer.add_summary(summary, step)
					print('Adding run metadata for', step)

					if FLAGS.trace:
						fetched_timeline = timeline.Timeline(run_metadata.step_stats)
						chrome_trace = fetched_timeline.generate_chrome_trace_format()
						with open(FLAGS.log_dir + '/' + run_tag + '/train/timeline_%d.json' % step, 'w') as f:
							f.write(chrome_trace)
				
				else:  # Record x summary
					summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, step))
					if step % FLAGS.train_summary_frequency == 0:
						train_writer.add_summary(summary, step)

	train_writer.close()
	test_writer.close()
	for i in task_writer:
		i.close()


def main(_):

	optimizers = {
		'sgd': tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate),
		'adam': tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
	}

	train(optimizers[FLAGS.optimizer])


if __name__ == '__main__':

	parser = make_argparse()
	FLAGS, unparsed = parser.parse_known_args()
	
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
