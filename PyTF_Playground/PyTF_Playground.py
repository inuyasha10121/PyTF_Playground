import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 0.002
N_NODES = [8]

TRAIN_MAX_ITER = 10000
TRAIN_ERR_THRESH = 1E-8

INPUT_SIZE = 3
OUTPUT_SIZE = 3

inputvals  = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]
targetvals = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]]
#targetvals = [[0], [0], [0], [1], [0], [1], [1], [0]]
testinputs  = [[0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1],
               [1, 0, 0], [0, 0, 0], [1, 1, 0]]
testtargets = [[0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]]
#testtargets = [[0], [1], [0], [1], [1], [0], [1], [0]]

x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name='y')

def build_neural_network(data):
	input_layer = {'weights': tf.Variable(tf.truncated_normal([INPUT_SIZE, N_NODES[0]])),
				   'biases': tf.Variable(tf.truncated_normal([N_NODES[0]]))}
	li = tf.add(tf.matmul(data, input_layer['weights']), input_layer['biases'])
	li = tf.nn.relu(li)

	hidden_layer_list = [li]
	for nodecount in range(len(N_NODES) - 1):
		hidden_layer = {'weights': tf.Variable(tf.truncated_normal([N_NODES[nodecount], N_NODES[nodecount + 1]])),
						'biases': tf.Variable(tf.truncated_normal([N_NODES[nodecount + 1]]))}
		lh = tf.add(tf.matmul(hidden_layer_list[len(hidden_layer_list) - 1], hidden_layer['weights']), hidden_layer['biases'])
		tf.nn.relu(lh)
		hidden_layer_list.append(lh)

	output_layer = {'weights': tf.Variable(tf.truncated_normal([N_NODES[len(N_NODES) - 1], OUTPUT_SIZE])),
				    'biases': tf.Variable(tf.truncated_normal([OUTPUT_SIZE]))}

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
	loss = tf.reduce_mean(sm)
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(TRAIN_MAX_ITER):
			optimizer.run(feed_dict={x:inputvals, y:targetvals})
			cost = sess.run(loss, feed_dict={x:inputvals, y:targetvals})
			if(i % 100 == 0):
				print("Step: %d, cost: %g"%(i, cost))
				if(cost < TRAIN_ERR_THRESH):
					break

		for i in range(len(testinputs)-1):
			res = sess.run(results, feed_dict={x: [testinputs[i]]})
			print("Input: ", testinputs[i])
			for j in range(OUTPUT_SIZE):
				print("(", testtargets[i][j], "):", res[0][j])

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