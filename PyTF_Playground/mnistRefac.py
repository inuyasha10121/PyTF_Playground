#Refactor of mnistExample to work more like bincounter

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Setup some parameters for the future
LEARNING_RATE = 0.001 #Set the learning rate (Default 0.001)

write_for_tensorboard = 1 #Decide if we want to write a log files for Tensorboard visualization
tensorboard_file = 'D:/TensorFlow/logs/' #Where the log files should be saved

#Setup input data
mnist = input_data.read_data_sets("D:/TensorFlow/mnist-data", one_hot=True)

#Define how many nodes are in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_epochs = 10

n_classes = 10 #There are 10 numbers to process, 0-9
batch_size = 100 #Check only 100 images at a time

#Setup inital TF stuff
sess = tf.Session()

#Setup placeholders for the data
x = tf.placeholder('float', [None, 784], name='x') 
y = tf.placeholder('float', name='y_')

#Define the model parameters
with tf.name_scope('hl1'):
	W_fc1 = tf.truncated_normal([784, n_nodes_hl1], mean=0.5, stddev=0.707)
	W_fc1 = tf.Variable(W_fc1, name='W_fc1')
	b_fc1 = tf.truncated_normal([n_nodes_hl1], mean=0.5, stddev=0.707)
	b_fc1 = tf.Variable(b_fc1, name='b_fc1')
	h_fc1 = tf.nn.relu(tf.add(tf.matmul(x, W_fc1), b_fc1))
	tf.summary.histogram('W_fc1_summary', W_fc1)
	tf.summary.histogram('b_fc1_summary', b_fc1)
	tf.summary.histogram('h_fc1_summary', h_fc1)

with tf.name_scope('hl2'):
	W_fc2 = tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], mean=0.5, stddev=0.707)
	W_fc2 = tf.Variable(W_fc2, name='W_fc2')
	b_fc2 = tf.truncated_normal([n_nodes_hl2], mean=0.5, stddev=0.707)
	b_fc2 = tf.Variable(b_fc2, name='b_fc2')
	h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
	tf.summary.histogram('W_fc2_summary', W_fc2)
	tf.summary.histogram('b_fc2_summary', b_fc2)
	tf.summary.histogram('h_fc2_summary', h_fc2)

with tf.name_scope('hl3'):
	W_fc3 = tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], mean=0.5, stddev=0.707)
	W_fc3 = tf.Variable(W_fc3, name='W_fc3')
	b_fc3 = tf.truncated_normal([n_nodes_hl3], mean=0.5, stddev=0.707)
	b_fc3 = tf.Variable(b_fc3, name='b_fc3')
	h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))
	tf.summary.histogram('W_fc3_summary', W_fc3)
	tf.summary.histogram('b_fc3_summary', b_fc3)
	tf.summary.histogram('h_fc3_summary', h_fc3)

with tf.name_scope('lout'):
	W_fco = tf.truncated_normal([n_nodes_hl3, n_classes], mean=0.5, stddev=0.707)
	W_fco = tf.Variable(W_fco, name='W_fco')
	b_fco = tf.truncated_normal([n_classes], mean=0.5, stddev=0.707)
	b_fco = tf.Variable(b_fco, name='b_fco')
	layer_out = tf.add(tf.matmul(h_fc3, W_fco), b_fco)
	tf.summary.histogram('W_fco_summary', W_fco)
	tf.summary.histogram('b_fco_summary', b_fco)
	tf.summary.histogram('layer_out_summary', layer_out)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_out, labels=y))
	tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

if write_for_tensorboard == 1:
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(tensorboard_file)
	writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
	epoch_cost = 0
	for i in range(int(mnist.train.num_examples/batch_size)):
		epoch_x, epoch_y = mnist.train.next_batch(batch_size)
		i, c = sess.run([optimizer, cross_entropy], feed_dict={x: epoch_x, y: epoch_y})
		epoch_cost += c
	print('Epoch', epoch, 'completed out of', n_epochs, 'Loss:', epoch_cost)


correct = tf.equal(tf.arg_max(layer_out, 1), tf.arg_max(y,1)) #Check to see if x and y are identical
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}, session=sess))