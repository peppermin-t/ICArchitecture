import tensorflow as tf
from tensorflow.contrib.layers import flatten


class LeNet:

    def __init__(self):
        """
        Define some basic parameters here
        """
        pass

    def net(self, feats):
        """
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        :param feats: input features
        :return: logits
        """
        # layer 1
        # TODO: construct the conv1
        conv1_W = self.init_weight([5, 5, 1, 6])
        conv1_b = self.init_bias([6])
        conv1_h = tf.nn.relu(self.conv2d(feats, conv1_W) + conv1_b)

        # layer 2
        # TODO: construct the pool1
        pool1_h = self.max_pool_2x2(conv1_h)

        # layer 3
        # TODO: construct the conv2
        conv2_W = self.init_weight([5, 5, 6, 16])
        conv2_b = self.init_bias([16])
        conv2_h = tf.nn.relu(self.conv2d(pool1_h, conv2_W) + conv2_b)

        # layer 4
        # TODO: construct the pool2
        pool2_h = self.max_pool_2x2(conv2_h)

        # self.pool2_h_flat = tf.reshape(self.pool2_h, [-1, 7*7*16])
        pool2_h_flat = flatten(pool2_h)

        # layer 5
        # TODO: construct the fc1
        fc1_W = self.init_weight([7*7*16, 120])
        fc1_b = self.init_bias([120])
        fc1_h = tf.nn.relu(tf.matmul(pool2_h_flat, fc1_W) + fc1_b)

        # layer 6
        # TODO: construct the fc2
        fc2_W = self.init_weight([120, 84])
        fc2_b = self.init_bias([84])
        fc2_h = tf.nn.relu(tf.matmul(fc1_h, fc2_W) + fc2_b)

        # layer output
        # TODO: construct the fc3
        fc3_W = self.init_weight([84, 10])
        fc3_b = self.init_bias([10])
        fc3_h = tf.nn.sigmoid(tf.matmul(fc2_h, fc3_W) + fc3_b)

        return fc3_h

    def forward(self, feats):
        """
        Forward the network
        """

        return self.net(feats)

    @staticmethod
    def init_weight(shape):
        """
        Init weight parameter.
        """
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    @staticmethod
    def init_bias(shape):
        """
        Init bias parameter.
        """
        b = tf.zeros(shape)
        return tf.Variable(b)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
