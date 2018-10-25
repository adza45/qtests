import tensorflow as tf
import pickle
from collections import deque
import numpy as np

class Agent():
	def __init__(self):
		self.ACTIONS = 5
		self.GAME = 'qtest'
		self.terminal = False
		self.s, self.readout, self.h_fc1 = self.createNetwork()
		self.LEARNING_RATE = 0.000001
		self.a = tf.placeholder("float", [None, self.ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
		self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
		self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)

		self.D = deque()

		self.a_t = np.zeros([self.ACTIONS])
		self.a_t[0] = 1
		self.action_index = 0
		self.total_reward = 0
		self.readout_t = 0
		self.sess = tf.InteractiveSession()
		# saving and loading networks
		self.saver = tf.train.Saver()
		self.sess.run(tf.initialize_all_variables())
		self.checkpoint = tf.train.get_checkpoint_state("saved_networks")
		self.INITIAL_EPSILON = 0.99
		self.epsilon = self.INITIAL_EPSILON
		self.t = 0
		self.REPLAY_MEMORY = 50000
		self.OBSERVE = 50000
		self.EXPLORE = 10000000
		self.FINAL_EPSILON = 0.1
		self.BATCH = 32
		self.GAMMA = 0.99


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def createNetwork(self):
		W_conv1 = self.weight_variable([8, 8, 4, 32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4, 4, 32, 64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3, 3, 64, 64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([1600, 512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512, self.ACTIONS])
		b_fc2 = self.bias_variable([self.ACTIONS])

		s = tf.placeholder("float", [None, 65, 65, 4])

		h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		readout = tf.matmul(h_fc1, W_fc2) + b_fc2

		return s, readout, h_fc1
