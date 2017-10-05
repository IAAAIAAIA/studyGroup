import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print("ay something JACQUE'SZ2")

# b = tf.Variable( tf.zeros((100, )) )
# w = tf.Variable( tf.random_uniform((784, 100), -1, 1) )

# x = tf.placeholder(tf.float32, (100, 784))

# h = tf.nn.relu(tf.matmul(x, w) + b)

# # print(tf.get_default_graph().get_operations())

# prediction = tf.nn.softmax(h)
# label = tf.placeholder(tf.float32, [100,10])

# cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)

# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# sess.run(h, {x: np.random.random(100, 784)})


# a = tf.Variable([[1,2,3]], dtype=tf.float32)
# b = tf.constant([[4,5,6]], dtype=tf.float32)
# d = tf.placeholder(dtype=tf.float32, )
# # c = tf.matmul(a, b, transpose_b=True)
# a = tf.exp(a)
# c = a * b * d

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print( sess.run(c,  feed_dict={d: 14}) )

# print(c)

results = []

for h in range(600, 784, 5):
    x = tf.placeholder(tf.float32, [None, 784])

    W1 = tf.Variable(tf.zeros([784, h]))
    b1 = tf.Variable(tf.zeros([h]))

    W2 = tf.Variable(tf.zeros([h, 10]))
    b2 = tf.Variable(tf.zeros([10]))

    x_1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    y = tf.nn.softmax(tf.matmul(x_1, W2) + b2)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("H:", h, "\tAcc:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

