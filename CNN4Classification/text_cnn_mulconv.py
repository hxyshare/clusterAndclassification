import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters,  weight, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(weight, name="W")
            # [batch_size,sequence_length,embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # [batch_size,sequence_length,embedding_size,1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #filter大小，纵，列，x，个数
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # [batch_size, sequence_length-filter_size+1, 1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # [batch_size,1,1,num_filters]  concate zhihou shi num_filters 每次还是pool一个值出来
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # [batch_size,num_filters]
                pooled = tf.reshape(pooled, [-1, num_filters])
                # [batch_size,num_filters,1]
                pooled = tf.expand_dims(pooled, -1)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # [batch_size,num_filters,len(filter_sizes)] concat zhihou
        self.h_pool = tf.concat(axis=2, values=pooled_outputs)
        ###
        # [batch_size,num_filters,len(filter_sizes),1]
        self.h_pool = tf.expand_dims(self.h_pool,-1)
        #self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("new_conv_layer"):
            ##
            filter_shape = [4, len(filter_sizes), 1, 1]
            ##
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            ###
            # [batch_size,num_filters-3,1,1]
            conv = tf.nn.conv2d(
                    self.h_pool,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            ###
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            ####
            # [batch_size,num_filters-3]
            self.h_pool_flat= tf.reshape(h,[-1,num_filters-3])


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters-3, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # [batch_size,num_classes]
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
