import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 0.25
N_NODES = [100]

TRAIN_MAX_ITER = 5000
TRAIN_ERR_THRESH = 1E-8

INPUT_SIZE = 1
OUTPUT_SIZE = 1

train_in = []
train_out = []
test_in = []
test_out = []

xtest = []
ytest = []

for i in range (1000):
	temp = random.random() * (2.0 * np.pi)
	train_in.append([temp])
	train_out.append([np.sin(temp) + np.cos(temp)])
for i in range(50):
	temp = random.random() * (2.0 * np.pi)
	xtest.append(temp)
xtest.sort()
for i in range(50):
	test_in.append([xtest[i]])
	test_out.append([np.sin(xtest[i]) + np.cos(xtest[i])])
	ytest.append(np.sin(xtest[i]) + np.cos(xtest[i]))
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
	
	'''
	nnsetup = build_neural_network(x)
	results = tf.sigmoid(nnsetup, name='results')
	output = tf.nn.softmax(nnsetup)
	loss = -tf.reduce_sum(y*tf.log(output))
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
	'''
	
	prediction = build_neural_network(x)
	results = tf.sigmoid(prediction, name='results')

	#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
	sm = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
	loss = tf.reduce_mean(tf.square(prediction - train_out))
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(TRAIN_MAX_ITER):
			optimizer.run(feed_dict={x:train_in, y:train_out})
			cost = sess.run(loss, feed_dict={x:train_in, y:train_out})
			if(i % 100 == 0):
				print("Step: %d, cost: %g"%(i, cost))
				if(cost < TRAIN_ERR_THRESH):
					break

		ycalc = prediction.eval({x:test_in}, sess)
	
		correctPlot = plt.plot(xtest, ytest)
		calcPlot = plt.plot(test_in, ycalc)
		plt.legend([correctPlot, calcPlot], ["corr","calc"])
		plt.show()
	'''
	prediction = build_neural_network(x)
	results = tf.sigmoid(prediction, name='results')
	cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y)) #Get the difference between the prediction and the known value
	#cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
	#cost = tf.reduce_sum(cost)

	optimizer = tf.train.RMSPropOptimizer(0.25, momentum=0.5).minimize(cost)
	#optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
	#Cycles of feed forward + back propagation

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) #Start the TF session

		for i in range(TRAIN_MAX_ITER + 1):
			train_error = cost.eval(feed_dict={x: inputvals, y:targetvals})
			print("Step %d, error %g"%(i, train_error))
			if train_error < TRAIN_ERR_THRESH:
				break
			sess.run(optimizer, feed_dict={x: inputvals, y: targetvals})		
		print("Done training!")

		print("INPUT:\tOUTPUT:\tEXPECTED:\tTEST:")
		for i in range(len(testinputs)-1):
			res = sess.run(results, feed_dict={x: [testinputs[i]]})
			testcond = "FAIL"
			if((round(res[0][0]) == testtargets[i][0]) and (round(res[0][0]) == testtargets[i][0]) and (round(res[0][0]) == testtargets[i][0])):
				testcond = "PASS"
			print('%i %i %i\t%i\t%i\t\t%s'%(testinputs[i][0],testinputs[i][1],testinputs[i][2], round(res[0][0]), testtargets[i][0], testcond))
	'''
train_neural_network(x)