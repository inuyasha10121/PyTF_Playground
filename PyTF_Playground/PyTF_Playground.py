import tensorflow as tf
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.interpolate as interp
#from tensorflow.examples.tutorials.mnist import input_data

TEST_RES = 50

LEARNING_RATE = 0.25
N_NODES = [50]

TRAIN_MAX_ITER = 10000
TRAIN_ERR_THRESH = 1E-3

INPUT_SIZE = 2
OUTPUT_SIZE = 1

train_in = []
train_out = []
test_in = []
test_out = []

xtest = []
ytest = []

for i in range (1000):
	temp = random.random() * (2.0 * np.pi)
	temp2 = random.random() * 2.0 * np.pi
	train_in.append([temp, temp2])
	train_out.append([np.sin(temp * np.cos(temp2))])
for i in range(50):
	temp = random.random() * (2.0 * np.pi)
	temp2 = random.random() * 2.0 * np.pi
	xtest.append([temp, temp2])
xtest.sort(key=lambda x: x[0])
for i in range(50):
	test_in.append(xtest[i])
	test_out.append([np.sin(xtest[i][0] * np.cos(xtest[i][1]))])
	ytest.append(np.sin(xtest[i][0] * np.cos(xtest[i][1])))
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
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	
	sess.run(tf.global_variables_initializer())
	for i in range(TRAIN_MAX_ITER):
		optimizer.run(feed_dict={x:train_in, y:train_out}, session=sess)
		cost = sess.run(loss, feed_dict={x:train_in, y:train_out})
		if(i % 100 == 0):
			print("Step: %d, cost: %g"%(i, cost))
			if(cost < TRAIN_ERR_THRESH):
				break

	testvals = prediction.eval({x:test_in}, sess)
	zvals = [row[0] for row in testvals]
	xvals = [row[0] for row in xtest]
	yvals = [row[1] for row in xtest]

	plt.plot(xvals, zvals)
	plt.plot(xvals, ytest)

	plt.show()
		
train_neural_network(x)