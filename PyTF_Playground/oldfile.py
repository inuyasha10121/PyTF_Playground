from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def test_func(posx, posy, chargex, chargey, chargeval):
	dist = math.sqrt((posx - chargex)**2 + (posy-chargey)**2)
	return chargeval / dist
	#return np.sin(posx * np.cos(posy))
	#return np.sin(a)

def error_func(a, b):
	return abs(a - b)

TRAIN_TOTAL_CASES = 100
TRAIN_TOTAL_POINTS = 100
TRAIN_CHARGE_POINTS = 1

TEST_TOTAL_CASES = 1
TEST_GRID_RES = 50
TEST_CHARGE_POINTS = 1

exclusiondist = 0.25

xrange = [-3.0, 3.0]
yrange = [-3.0, 3.0]


LEARNING_RATE = 0.1
N_NODES = [50]

TRAIN_MAX_ITER = 10000
TRAIN_ERR_THRESH = 1E-3

INPUT_SIZE = 3 #[X, Y, Charge]
OUTPUT_SIZE = 1

train_in = []
train_out = []
test_in = []
test_out = []

plotx = []
ploty = []
plotz = []

xscope = xrange[1] - xrange[0]
yscope = yrange[1] - yrange[0]

xstep = xscope / TEST_GRID_RES
ystep = yscope / TEST_GRID_RES

#Start generating training data
for case in range(TRAIN_TOTAL_CASES):
	train_in.append([])
	train_out.append([])
	traincharges = []
	#Add the charge point data
	for i in range(TRAIN_CHARGE_POINTS):
		xpos = xrange[0] + (xscope * random.random())
		ypos = yrange[0] + (yscope * random.random())
		chargestr = 1.0
		train_in[case].append([xpos, ypos, chargestr])
		traincharges.append([xpos, ypos, chargestr])
		train_out[case].append([0.0])
	for i in range(TRAIN_TOTAL_POINTS - TRAIN_CHARGE_POINTS):
		xpos = xrange[0] + (xscope * random.random())
		ypos = yrange[0] + (yscope * random.random())
		train_in[case].append([xpos, ypos, 0.0])
		exclude = False
		for j in range(TRAIN_CHARGE_POINTS):
			dist = math.sqrt((xpos - traincharges[j][0])**2 + (ypos - traincharges[j][1])**2)
			if dist < exclusiondist:
				exclude = True
		if exclude:
			train_out[case].append([0.0])
		else:
			train_out[case].append(test_func(xpos, ypos,traincharges[0][0], traincharges[0][1], traincharges[0][2])) #TODO: This only works for 1 charge.  



'''
for j in range(TEST_RES):
	plotx.append([])
	ploty.append([])
	plotz.append([])
	for i in range(TEST_RES):
		xpos = xrange[0] + (i * xstep)
		ypos = yrange[0] + (j * ystep)
		plotx[j].append(xpos)
		ploty[j].append(ypos)
		test_in.append([xpos, ypos])
		
		dist = math.sqrt((xpos - chargex)**2 + (ypos-chargex)**2)
		if dist > exclusiondist:
			plotz[j].append(test_func(xpos, ypos, chargex, chargey, chargeval))
			test_out.append([test_func(xpos, ypos, chargex, chargey, chargeval)])
		else:
			plotz[j].append(0.0)
			test_out.append([0.0])



for i in range (TRAIN_RES):
	xpos = xrange[0] + (random.random() * xscope)
	ypos = yrange[0] + (random.random() * yscope)
	train_in.append([xpos, ypos])

	dist = math.sqrt((xpos - chargex)**2 + (ypos-chargex)**2)
	if dist > exclusiondist:
		train_out.append([test_func(xpos, ypos, chargex, chargey, chargeval)])
	else:
		train_out.append([0.0])

	

'''

#Start all the Tensorflow stuff
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE], name='x')
y = tf.placeholder(tf.float32, shape=[None, None, OUTPUT_SIZE], name='y')

def build_neural_network(data):
	#input_layer = {'weights': tf.Variable(tf.random_normal([N_NODES[0], INPUT_SIZE, N_NODES[0]])),
	#			   'biases': tf.Variable(tf.random_normal([N_NODES[0], N_NODES[0]]))}
	flatdata = tf.reshape(data, [None, INPUT_SIZE])
	input_layer = {'weights': tf.Variable(tf.random_normal([INPUT_SIZE, N_NODES[0]])),
				   'biases': tf.Variable(tf.random_normal([N_NODES[0], N_NODES[0]]))}
	li = tf.add(tf.matmul(data, input_layer['weights']), input_layer['biases'])
	li = tf.nn.relu(li)

	hidden_layer_list = [li]
	for nodecount in range(len(N_NODES) - 1):
		hidden_layer = {'weights': tf.Variable(tf.random_normal([N_NODES[nodecount + 1], N_NODES[nodecount], N_NODES[nodecount + 1]])),
						'biases': tf.Variable(tf.random_normal([N_NODES[nodecount + 1], N_NODES[nodecount + 1]]))}
		lh = tf.add(tf.matmul(hidden_layer_list[len(hidden_layer_list) - 1], hidden_layer['weights']), hidden_layer['biases'])
		tf.nn.relu(lh)
		hidden_layer_list.append(lh)

	output_layer = {'weights': tf.Variable(tf.random_normal([TRAIN_TOTAL_CASES, N_NODES[len(N_NODES) - 1], OUTPUT_SIZE])),
					'biases': tf.Variable(tf.random_normal([OUTPUT_SIZE, OUTPUT_SIZE]))}

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
			calcz[j].append(testvals[(j * TEST_RES) + i][0])
	
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