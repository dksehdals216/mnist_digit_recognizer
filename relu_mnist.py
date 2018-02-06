import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#images are 28x28, (784px total)
#pixel values are int from 0~255 indicating lightness or darkness

#train dataset has 785 col. first col is label, digit drawn by the user
#Rest of the col contain pixel-values of associated image
#each pixel col has name like pexelx where x is from 0~783 inclusive.
#

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#params
training_epochs = 1000
batch_size = 100
learning_rate = 0.01
node_n = 256

nb_classes = 10

#28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0~9 digits, 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

#layer1
W1 = tf.Variable(tf.random_normal([784, node_n]))
b1 = tf.Variable(tf.random_normal([node_n]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

#layer2
W2 = tf.Variable(tf.random_normal([node_n, node_n]))
b2 = tf.Variable(tf.random_normal([node_n]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

#layer3
W3 = tf.Variable(tf.random_normal([node_n,10]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
#Hypothesis using softmax
hypothesis = (tf.matmul(layer2, W3) + b3)

#cross-entropy of onehot Y
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# testing
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    #init TF variable
    sess.run(tf.global_variables_initializer())
    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size )

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    #Creates batches of 100 because there is no need for all data to be on memory
    batch_xs, batch_ys = mnist.train.next_batch(100)

    print("acc: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    #predict 1 sample
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow( 
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
