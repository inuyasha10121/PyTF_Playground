from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def test_func(in_x, in_y, ori_x, ori_y):
	return (np.sin(in_x - ori_x) + np.cos(in_y - ori_y))

def error_func(a, b):
	#return abs(a - b)
	return (a-b)**2

######################################################################### SIMLUATION PARAMETERS #########################################################################
#Dim 1
TRAIN_CASE_NUM = 1000
TEST_CASE_NUM = 1

#Dim 2
TRAIN_SET_NUM = 500 #Number of random points to generate for test data
TEST_SET_NUM = 50 #Grid resolution for test data map

#Dim 3
INPUT_SIZE = 2 #[x,y]
OUTPUT_SIZE = 1 #[Result]

xrange = [-3.0, 3.0] #Overall X range of search space
yrange = [-3.0, 3.0] #Overall Y range of search space

LEARNING_RATE = 0.1
N_NODES = [100] #Neural network topology

TRAIN_MAX_ITER = 10000 #Maximum iterations for training cycle
TRAIN_ERR_THRESH = 1E-3 #Error threshold for training cycle
######################################################################### GENERATE SIMULATION DATA #########################################################################
#Scope for calculation space
xscope = xrange[1] - xrange[0]
yscope = yrange[1] - yrange[0]

#Step resolution for test grid generation
xstep = xscope / TEST_SET_NUM
ystep = yscope / TEST_SET_NUM

train_in = []
train_out = []
test_in = []
test_out = []

plotx = []
ploty = []
plotz = []
'''
#Train set generation
for case in range(TRAIN_CASE_NUM):
	print("Populating training: " + str(case + 1) + "/" + str(TRAIN_CASE_NUM))
	train_in.append([])
	train_out.append([])
	ori_x = random.random() * 5.0
	ori_y = random.random() * 5.0
	train_in[case].append([ori_x, ori_y])
	train_out[case].append([test_func(ori_x, ori_y, ori_x, ori_y)])
	for set in range(TRAIN_SET_NUM - 1):
		in_x = xrange[0] + (random.random() * xscope)
		in_y = yrange[0] + (random.random() * yscope)
		train_in[case].append([in_x, in_y])
		train_out[case].append([test_func(in_x, in_y, ori_x, ori_y)])
'''
for case in range(TEST_CASE_NUM):
	print("Populating test: " + str(case + 1) + "/" + str(TEST_CASE_NUM))
	plotx.append([])
	ploty.append([])
	plotz.append([])
	test_in.append([])
	test_out.append([])
	#ori_x = random.random() * 5.0
	#ori_y = random.random() * 5.0
	ori_x = 0.0
	ori_y = 0.0
	for j in range(TEST_SET_NUM):
		plotx[case].append([])
		ploty[case].append([])
		plotz[case].append([])
		for i in range(TEST_SET_NUM):
			in_x = xrange[0] + (i * xstep)
			in_y = yrange[0] + (j * ystep)
			res = test_func(in_x, in_y, ori_x, ori_y)

			plotx[case][j].append(in_x)
			ploty[case][j].append(in_y)
			plotz[case][j].append(res)

			test_in[case].append([in_x, in_y])
			test_out[case].append([res])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(plotx[0], ploty[0], plotz[0], rstride = 5, cstride = 5, color = 'g', label = 'Corr')
ax.legend()
plt.show()
exit()
'''
#Scope for calculation space
xscope = xrange[1] - xrange[0]
yscope = yrange[1] - yrange[0]


#Step resolution for test grid generation
xstep = xscope / TEST_RES
ystep = yscope / TEST_RES

particles.append([2.0, 2.0, 1.0])
particles.append([-2.0, -2.0, 1.0])
#Generate training dataset
for i in range (TRAIN_RES):
	xpos = xrange[0] + (random.random() * xscope)
	ypos = yrange[0] + (random.random() * yscope)
	train_in.append([xpos, ypos, 0.0])
	train_out.append([multichargefieldstrength(exclusiondist, xpos, ypos, particles)])

#Generate test grid for later validation
for j in range(TEST_RES):
	plotx.append([])
	ploty.append([])
	plotz.append([])
	for i in range(TEST_RES):
		xpos = xrange[0] + (i * xstep)
		ypos = yrange[0] + (j * ystep)
		plotx[j].append(xpos)
		ploty[j].append(ypos)
		test_in.append([xpos, ypos, 0.0])

		plotz[j].append(multichargefieldstrength(exclusiondist, xpos, ypos, particles))
		test_out.append([multichargefieldstrength(exclusiondist, xpos, ypos, particles)])
'''
######################################################################### MACHINE LEARNING SECTION ########################################################################

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
	for case in range(TEST_CASE_NUM):
		calcz.append([])
		for j in range(TEST_SET_NUM):
			calcz[case].append([])
			for i in range(TEST_SET_NUM):
				calcz[case][j].append(testvals[case][(j * TEST_RES) + i][0])

			
train_neural_network(x)