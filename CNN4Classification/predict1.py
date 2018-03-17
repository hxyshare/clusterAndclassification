# coding=utf-8
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
import random
from text_cnn_mulconv import TextCNN
import time
import datetime

ISOTIMEFORMAT='%Y-%m-%d %X'
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.005, "learning_rate (default: 0.001)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("dev_batch_size", 200, "Batch Size (default: 1000)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def pad_sentences(sentence, sequence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 20
    sentence_list = sentence.strip().split(' ')
    if 200 > len(sentence_list):
        num_padding = sequence_length - len(sentence_list)
        padding_word = "<PAD/>"
        new_sentence = sentence_list + [padding_word] * num_padding
    else:
        new_sentence = sentence_list[0:sequence_length]
    return new_sentence


def load_wv_model(word2vec_path, vocabulary_inv):
    # build word vector matrix W
    model = Word2Vec.load(word2vec_path)
    # W = [model[v.decode('utf-8')]for v in vocabulary_inv]
    W = []
    for v in vocabulary_inv:
        try:
            W.append(np.float32(model[v]))
        except Exception:
            l = []
            for i in range(FLAGS.embedding_dim):
                l.append(random.uniform(1, -1))
            W.append(np.float32(np.array(l)))
    return np.array(W)


def load_cnn(sl, W,cnn_path, vocabulary_length):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=sl,
                num_classes=10,
                vocab_size=vocabulary_length,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
		weight =W,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            saver = tf.train.Saver()
            saver.restore(sess, cnn_path)
            print ('--------CNN model has been loaded--------')

    def pre_step(x, y):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x,
          cnn.input_y: y,
          cnn.dropout_keep_prob: 1.0
        }
        scores = sess.run([cnn.scores], feed_dict)
        return scores
    return pre_step


def predict(sentence, sequence_length, vocabulary_invmap, cnn_model, labels):
    #print datetime.datetime.now().isoformat()+'in predict'
    sentences_padded = pad_sentences(sentence, sequence_length)
    #print datetime.datetime.now().isoformat()+'pad sen'

    xx1 = []
    for w in sentences_padded:
        if str(w) in vocabulary_invmap:
            xx1.append(vocabulary_invmap[str(w)])
        else:
            xx1.append(vocabulary_invmap['<PAD/>'])
    #print datetime.datetime.now().isoformat() + 'vocab'
    xx1 = np.array([xx1])
    #print datetime.datetime.now().isoformat() + 'xx1'
    y1 = np.array([labels])
    #print datetime.datetime.now().isoformat() + 'y1'
    sc = cnn_model(xx1, y1)
    #print datetime.datetime.now().isoformat() + 'CNN model'
    
    return sc

