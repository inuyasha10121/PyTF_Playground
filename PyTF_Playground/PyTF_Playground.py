from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def test_func(posx, posy, chargex, chargey, chargeval):
	dist = math.sqrt((posx - chargex)**2 + (posy-chargey)**2)
	return chargeval / dist

#Function for calculating the field strength at a point from multiple charged particles
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

'''
chargex = 1.0
chargey = 1.0
'''
NUM_CHARGE_PARTICLES = 1 #How many charged particles to include in the simulations
chargeval = 1.0 #Charge value for all charged particles
exclusiondist = 0.25 #Distance around charged particles to exclude test points

xrange = [-3.0, 3.0] #Overall X range of search space
yrange = [-3.0, 3.0] #Overall Y range of search space

TRAIN_RES = 1000 #Number of random points to generate for test data
TEST_RES = 50 #Grid resolution for test data map

LEARNING_RATE = 0.1
N_NODES = [100] #Neural network topology

TRAIN_MAX_ITER = 10000 #Maximum iterations for training cycle
TRAIN_ERR_THRESH = 1E-3 #Error threshold for training cycle

INPUT_SIZE = 3 #[X coord, Y coord, Field value]
#NOTE: Charge 0 indicates a field point.
OUTPUT_SIZE = 1 #[Field Strength]

######################################################################### GENERATE SIMULATION DATA #########################################################################

particles = []
train_in = []
train_out = []
test_in = []
test_out = []

plotx = []
ploty = []
plotz = []

#Scope for calculation space
xscope = xrange[1] - xrange[0]
yscope = yrange[1] - yrange[0]


#Step resolution for test grid generation
xstep = xscope / TEST_RES
ystep = yscope / TEST_RES

'''
for i in range(NUM_CHARGE_PARTICLES):
	xpos = xrange[0] + (random.random() * xscope)
	ypos = yrange[0] + (random.random() * yscope)
	charge = chargeval
	particles.append([xpos, ypos, charge])
'''
particles.append([1.0, 1.0, 1.0])
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