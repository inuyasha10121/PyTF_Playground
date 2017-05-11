from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def test_func(a, b):
	return np.sin(a * np.cos(b))
	#return np.sin(a)

def error_func(a, b):
	return abs(a - b)

TRAIN_RES = 1000
TEST_RES = 50

LEARNING_RATE = 0.1
N_NODES = [100]

TRAIN_MAX_ITER = 10000
TRAIN_ERR_THRESH = 1E-3

INPUT_SIZE = 2
OUTPUT_SIZE = 1

train_in = []
train_out = []
test_in = []
test_out = []

plotx = []
ploty = []
plotz = []


test_step = (2.0 * np.pi) / TEST_RES
for j in range(TEST_RES):
	plotx.append([])
	ploty.append([])
	plotz.append([])
	for i in range(TEST_RES):
		plotx[j].append(i * test_step)
		ploty[j].append(j * test_step)
		plotz[j].append(test_func(i * test_step, j * test_step))

for i in range (TRAIN_RES):
	temp = random.random() * (2.0 * np.pi)
	temp2 = random.random() * (2.0 * np.pi)
	train_in.append([temp, temp2])
	train_out.append([test_func(temp, temp2)])

for j in range(TEST_RES):
	for i in range(TEST_RES):
		test_in.append([plotx[i][j], ploty[i][j]])
		test_out.append([test_func(plotx[i][j], ploty[i][j])])

'''
train_in  = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
			  [1, 1, 0], [1, 1, 1]]
train_out = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]]
#train_out = [[0], [0], [0], [1], [0], [1], [1], [0]]
test_in  = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1],
			   [1, 0, 0], [0, 0, 0], [1, 1, 0]]
test_out = [[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]]
#test_out = [[0], [1], [0], [1], [1], [0], [1], [0]]
'''

#Start all the Tensorflow stuff
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name='y')

def build_neural_network(data):
	input_layer = {'weights': tf.Variable(tf.random_normal([INPUT_SIZE, N_NODES[0]])),
				   'biases': tf.Variable(tf.random_normal([N_NODES[0]]))}
	li = tf.add(tf.matmul(data, input_layer['weights']), input_layer['biases'])
	li = tf.nn.relu(li)

	hidden_layer_list = [li]
	for nodecount in range(len(N_NODES) - 1):
		hidden_layer = {'weights': tf.Variable(tf.random_normal([N_NODES[nodecount], N_NODES[nodecount + 1]])),
						'biases': tf.Variable(tf.random_normal([N_NODES[nodecount + 1]]))}
		lh = tf.add(tf.matmul(hidden_layer_list[len(hidden_layer_list) - 1], hidden_layer['weights']), hidden_layer['biases'])
		tf.nn.relu(lh)
		hidden_layer_list.append(lh)

	output_layer = {'weights': tf.Variable(tf.random_normal([N_NODES[len(N_NODES) - 1], OUTPUT_SIZE])),
					'biases': tf.Variable(tf.random_normal([OUTPUT_SIZE]))}

	output = tf.add(tf.matmul(hidden_layer_list[len(hidden_layer_list) - 1], output_layer['weights']), output_layer['biases'])

	return output

def train_neural_network(x):
		
	prediction = build_neural_network(x)
	results = tf.sigmoid(prediction, name='results')
	
	sess = tf.Session()

	#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
	#sm = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
	#sm1 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
	loss = tf.reduce_mean(tf.abs((prediction - train_out)))
	optimizer = tf.train.AdamOptimizer().minimize(loss)
	#optimizer = tf.train.RMSPropOptimizer(0.1).minimize(loss)
	
	sess.run(tf.global_variables_initializer())
	for i in range(TRAIN_MAX_ITER):
		optimizer.run(feed_dict={x:train_in, y:train_out}, session=sess)
		cost = sess.run(loss, feed_dict={x:train_in, y:train_out})
		if(i % 100 == 0):
			print("Step: %d, cost: %g"%(i, cost))
			if(cost < TRAIN_ERR_THRESH):
				break

	testvals = prediction.eval({x:test_in}, sess)

	calcz = []
	for j in range(TEST_RES):
		calcz.append([])
		for i in range(TEST_RES):
			calcz[j].append(testvals[(i * TEST_RES) + j][0])
	
	#Calculate average error
	avgerror = 0.0
	for j in range(TEST_RES):
		for i in range(TEST_RES):
			cz = calcz[i][j]
			pz = plotz[i][j]
			currerror = error_func(cz, pz)

			avgerror += currerror
	avgerror /= (TEST_RES * TEST_RES)
	print("Average test distance error: %f"%(avgerror))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(plotx, ploty, plotz, rstride=5, cstride=5, color='g', label='Corr')
	ax.plot_wireframe(plotx, ploty, calcz, rstride=5, cstride=5, color='r', label='Calc')
	ax.legend()
	plt.show()
		
train_neural_network(x)