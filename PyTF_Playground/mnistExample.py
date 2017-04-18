#mnist digit recognition example tutorial from sentdex
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Setup some parameters for the future
LEARNING_RATE = 0.001 #Set the learning rate (Default 0.001)

#Setup input data
mnist = input_data.read_data_sets("D:/TensorFlow/mnist-data", one_hot=True)

#Define how many nodes are in each hidden layer
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 500

n_classes = 10 #There are 10 numbers to process, 0-9
batch_size = 100 #Check only 100 images at a time

#Setup placeholders for the data
#784 comes from the fact the 28x28 images are flattened into a 1D matrix
#shape=[height,width]
x = tf.placeholder('float', [None, 784]) 
y = tf.placeholder('float', )

#Define the model parameters
def neural_network_model(data):
	#Setup weights and biases for the hidden layer
	#Weights multiply the input data values, biases are added after
	#If biases weren't there, then some neurons would never activate
	#TODO: Try to do ALL of this into a for loop for elegance
	hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	hidden_layer_out = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}

	#(input_data * weights) + biases
	#Establish layer models
	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1) #Activation function (threshold function)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2) #Activation function (threshold function)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3) #Activation function (threshold function)

	output = tf.matmul(l3, hidden_layer_out['weights']) + hidden_layer_out['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x) #Pass data through the neural network
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #Get the difference between the prediction and the known value

	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost) #Stocastic gradient decent

	#Cycles of feed forward + back propagation
	nEpochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer()) #Start the TF session

		for epoch in range(nEpochs):
			epoch_cost = 0

			for i in range(int(mnist.train.num_examples/batch_size)): #Iterate through all the data, batch by batch
				epoch_x, epoch_y = mnist.train.next_batch(batch_size) #Populate the batch of data
				i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) #Optimize the cost based on the data by changing the weights and biases
				epoch_cost += c
			print('Epoch ', epoch, ' completed out of ', nEpochs, ' Loss: ', epoch_cost)

		correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y,1)) #Check to see if x and y are identical
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)