import tensorflow as tf
import numpy as np

#Parameters for the calculation
NUM_TRAINING_SETS = 10000
NUM_TEST_SETS = 1000

LEARNING_RATE = 0.001
ERR_THRESH = 1E-5
MAX_ITER = 10000

N_NODES_H1 = 500
N_NODES_H2 = 500
N_NODES_H3 = 500

#Populate the training set with x,y,z data
print("Populating training input...")

training_input = [[0 for x in range(3)] for y in range(NUM_TRAINING_SETS)]
training_output = [[1] for y in range(NUM_TRAINING_SETS)]

for i in range(NUM_TRAINING_SETS):
	training_input[i] = [np.random.uniform(-30.0, 30.0), np.random.uniform(-30.0, 30.0), np.random.uniform(-30.0, 30.0)]
	training_output[i] = [np.sin(training_input[i][0]) + np.cos(training_input[i][1]) + np.tan(training_input[i][1])]

test_input = [[0 for x in range(3)] for y in range(NUM_TEST_SETS)]
test_output = [[1] for y in range(NUM_TEST_SETS)]

for i in range(NUM_TEST_SETS):
	test_input[i] = [np.random.uniform(-30.0, 30.0), np.random.uniform(-30.0, 30.0), np.random.uniform(-30.0, 30.0)]
	test_output[i] = [np.sin(training_input[i][0]) + np.cos(training_input[i][1]) + np.tan(training_input[i][1])]

print("Done!")

#Setup initial TF stuff
x = tf.placeholder('float', shape=[None, 3], name='x')
y = tf.placeholder('float', shape=[None, 1], name='y_')

with tf.name_scope('layer1'):
	W_fc1 = tf.truncated_normal([NUM_TRAINING_SETS, N_NODES_H1], mean=0.5, stddev=0.707)
	W_fc1 = tf.Variable(W_fc1, name='W_fc1')

	b_fc1 = tf.truncated_normal([N_NODES_H1], mean=0.5, stddev=0.707)
	b_fc1 = tf.Variable(b_fc1, name='b_fc1')

	h_fc1 = tf.nn.relu(tf.add(tf.matmul(x, W_fc1), b_fc1))


