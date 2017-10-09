from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def test_func(in_x, in_y, ori_x, ori_y):
	return (np.sin(in_x - ori_x) + np.cos(in_y - ori_y))

#excdist: Exclusion radius around each charged particle
#posx: Field point x position
#posy: Field point y position
#particles: List of charged particles( [[chargepos_x, chargepos_y, charge_str],...] )
def multichargefieldstrength(excdist, posx, posy, particles):
	fieldx = 0.0
	fieldy = 0.0
	for particle in particles: #Cycle through each particle
		distx = posx - particle[0]
		disty = posy - particle[1]
		dist = math.sqrt(distx**2 + disty**2) #Get the distance from the point to particle in question
		if(dist < excdist): #If we are ever in an exclusion zone, return 0 to avoid massive field strengths
			return 0.0
		angle = np.arctan(disty/distx) #Calculate the key angle of the field vector
		if(distx < 0.0): #If we are in the negative x, we need to tweak the angle
			angle += np.pi
		field = particle[2] / dist #Calculate the field strength of the vector (NOTE: This might need to be made negative, depending on how the math works out)
		#Add the vector components to the overall field strength components
		fieldx += np.sin(angle) * field
		fieldy += np.cos(angle) * field
	totalfield = math.sqrt(fieldx**2 + fieldy**2)
	return totalfield

def error_func(a, b):
	#return abs(a - b)
	return (a-b)**2

######################################################################### SIMLUATION PARAMETERS #########################################################################

### POINT CHARGE PARAMETERS ###
NUM_POTENTIAL_CHARGES = 10
NUM_CHARGES = 1
CHARGE_VALUE = 1
EXCLUSION_DIST = 0.25

### MAP PARAMETERS ###
xrange = [-3.0, 3.0] #Overall X range of search space
yrange = [-3.0, 3.0] #Overall Y range of search space

### DATA STRUCTURE PARAMETERS ###
#Dim 1 (Number of maps to test)
TRAIN_CASE_NUM = 10
TEST_CASE_NUM = 1

#Dim 2 (Points per map)
TEST_SET_NUM = 50 #Grid resolution for test data map
TRAIN_SET_NUM = TEST_SET_NUM * TEST_SET_NUM

#Dim 3 (Info contained in each point)
INPUT_SIZE = 3 #[x,y,charge]
OUTPUT_SIZE = 1 #[field value]

### NEURAL NETWORK PARAMETERS ###
LEARNING_RATE = 0.1
N_NODES = [100, 100] #Neural network topology

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

#Generate training set information
for case in range(TRAIN_CASE_NUM):
	print("Populating training: " + str(case + 1) + "/" + str(TRAIN_CASE_NUM))
	#Add new case entry
	train_in.append([])
	train_out.append([])
	#Generate new charges list
	point_charges = []
	for pc in range(NUM_CHARGES):
		pc_x = xrange[0] + (random.random() * xscope)
		pc_y = yrange[0] + (random.random() * yscope)
		point_charges.append([pc_x, pc_y, CHARGE_VALUE])
		train_in[case].append([pc_x, pc_y, CHARGE_VALUE])
		train_out[case].append([0.0])
	#Put in zero-fill for missing charges
	for mc in range(NUM_POTENTIAL_CHARGES - NUM_CHARGES):
		train_in[case].append([0.0, 0.0, 0.0])
		train_out[case].append([0.0])
	#Populate the actual sample map
	for i in range(TRAIN_SET_NUM):
		fp_x = xrange[0] + (random.random() * xscope)
		fp_y = yrange[0] + (random.random() * yscope)
		fval = multichargefieldstrength(EXCLUSION_DIST, fp_x, fp_y, point_charges)
		train_in[case].append([fp_x, fp_y, 0.0])
		train_out[case].append([fval])

#Generate testing set information
for case in range(TEST_CASE_NUM):
	print("Populating test: " + str(case + 1) + "/" + str(TEST_CASE_NUM))
	#Add new case entry
	plotx.append([])
	ploty.append([])
	plotz.append([])
	test_in.append([])
	test_out.append([])
	#Generate new chages list
	point_charges = []
	for pc in range(NUM_CHARGES):
		pc_x = xrange[0] + (random.random() * xscope)
		pc_y = yrange[0] + (random.random() * yscope)
		point_charges.append([pc_x, pc_y, CHARGE_VALUE])
		test_in[case].append([pc_x, pc_y, CHARGE_VALUE])
		test_out[case].append([0.0])
	#Put in zero-fill for missing charges
	for mc in range(NUM_POTENTIAL_CHARGES - NUM_CHARGES):
		test_in[case].append([0.0, 0.0, 0.0])
		test_out[case].append([0.0])

	#Populate map grid
	for j in range(TEST_SET_NUM):
		plotx[case].append([])
		ploty[case].append([])
		plotz[case].append([])
		for i in range(TEST_SET_NUM):
			fp_x = xrange[0] + (i * xstep)
			fp_y = yrange[0] + (j * ystep)
			fval = multichargefieldstrength(EXCLUSION_DIST, fp_x, fp_y, point_charges)

			plotx[case][j].append(fp_x)
			ploty[case][j].append(fp_y)
			plotz[case][j].append(fval)

			test_in[case].append([fp_x, fp_y, 0.0])
			test_out[case].append([fval])

######################################################################### MACHINE LEARNING SECTION ########################################################################
#Start all the Tensorflow stuff
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE], name='x')
y = tf.placeholder(tf.float32, shape=[None, None, OUTPUT_SIZE], name='y')

def build_neural_network(data):
	input_layer = {'weights': tf.Variable(tf.random_normal([INPUT_SIZE, N_NODES[0], N_NODES[0]])),
				   'biases': tf.Variable(tf.random_normal([N_NODES[0], N_NODES[0]]))}
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
	
	#Harvest predicted Z values
	'''
	calcz = []
	for case in range(TEST_CASE_NUM):
		calcz.append([])
		for j in range(TEST_SET_NUM):
			calcz[case].append([])
			for i in range(TEST_SET_NUM):
				calcz[case][j].append(testvals[case][(j * TEST_RES) + i][0])
	'''
	#Display plots
	for i in range(TEST_CASE_NUM):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_wireframe(plotx[i][NUM_CHARGES:(NUM_CHARGES + TRAIN_SET_NUM)], ploty[i][NUM_CHARGES:(NUM_CHARGES + TRAIN_SET_NUM)], plotz[i][NUM_CHARGES:(NUM_CHARGES + TRAIN_SET_NUM)], rstride = 5, cstride = 5, color = 'g', label = ('Corr[' + str(i) + ']'))
		ax.legend()
		plt.show()
			
train_neural_network(x)