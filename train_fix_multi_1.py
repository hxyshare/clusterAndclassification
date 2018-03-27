#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers_fix as data_helpers
from text_cnn_mulconv import TextCNN
import cPickle as pkl
import sys
import random
import gensim
import argparse
# Parameters
# ==================================================
# Model Hyperparameters



    # Data Preparatopn
    # ==================================================
def train():
    # Load data
    print("Loading data...")
    if not os.path.exists('./data_no_sw.pkl'):
        x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
    else:
        with open('./data_no_sw.pkl', 'rb') as f:
            loaded_data = pkl.load(f)
            train_data, dev_data, vocabulary ,vocabulary_inv = \
                tuple(loaded_data[k] for k in
                      ['train_data', 'dev_data', 'vocabulary', 'vocabulary_inv'])

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))
    print("Train/Dev split: {:d}/{:d}".format(len(train_data), len(dev_data)))
    # build word vector matrix W
    print(len(train_data[2][0]))
    vocabu_len = len(vocabulary_inv)
    w2v_model = gensim.models.Word2Vec.load('./meituanw2v.model')
    # W = [model[v.decode('utf-8')]for v in vocabulary_inv]
    #相当于我在初始化，权重的时候，使用了已经训练的词限量
    W = []
    for v in vocabulary_inv:
        try:
            W.append(np.float32(w2v_model[v]))
        except Exception as ex:    
            l = []
            for i in range(FLAGS.embedding_dim):
                l.append(random.uniform(-0.25,0.25 ))
            W.append(np.float32(np.array(l)))
    W = np.array(W)

    print(W.shape)
    # Training
    # ==================================================

    with tf.Graph().as_default():

        # new session with log_device_placement and set True.
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("cnn initialize!")
            cnn = TextCNN(
                sequence_length=20,
                num_classes=FLAGS.num_classes,
                vocab_size=vocabu_len,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                weight=W,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            print("cnn built!")
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            out_dir = log_dir + "/saved_model"
            print ("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            ckpt = tf.train.get_checkpoint_state('/data1/sina_dw/liping18/raw_cnn1.0_3.1/runs/1491055112345/checkpoints/')
            if ckpt and ckpt.model_checkpoint_path:
                print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                train_summary_writer.add_summary(summaries, step)
                return loss, accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                if writer:
                    writer.add_summary(summaries, step)
                return loss,accuracy
            print("\n")

            # Training loop. For each epoch...
            accuracy_all, loss_all = 0.0, 0.0
            for epoch in range(FLAGS.num_epochs):

                random.shuffle(train_data)
                num_batches_per_epoch = int(len(train_data) / FLAGS.batch_size) + 1
                start_index=0
                end_index=FLAGS.batch_size

                #for each batch
                for batch in range(num_batches_per_epoch):
                    if(end_index<=len(train_data)):
                        x_batch_list,y_batch_list= zip(*train_data[start_index:end_index])
                    else:
                        x_batch_list,y_batch_list = zip(*train_data[start_index: len(train_data)])

                    x_batch=np.array(x_batch_list)
                    y_batch=np.array(y_batch_list)

                    start_index+=FLAGS.batch_size
                    end_index+=FLAGS.batch_size
                    if end_index>len(train_data):
                        break

                    try:
                        step_loss, step_accuracy = train_step(x_batch, y_batch)
                        current_step = tf.train.global_step(sess, global_step)
                        loss_all += step_loss
                        accuracy_all += step_accuracy
                        if current_step % FLAGS.print_every == 0:
                             #print loss and accuracy
                            time_str = datetime.datetime.now().isoformat()
                            print(time_str+" epoch "+str(epoch)+" step "+str(current_step)+" loss "+str(loss_all/FLAGS.print_every)+" acc "+str(accuracy_all/FLAGS.print_every))
                            sys.stdout.flush()
                            loss_all, accuracy_all = 0.0, 0.0

                        if current_step % FLAGS.evaluate_every == 0:
                            #develop
                            num_batches_per_epoch_dev = int(len(dev_data) / FLAGS.dev_batch_size) + 1
                            start_index_dev = 0
                            end_index_dev = FLAGS.dev_batch_size
                            loss_dev, accuracy_dev = 0.0, 0.0
                            count = 0
                            for dev_batch in range(num_batches_per_epoch_dev):
                               
                                if (end_index_dev <= len(dev_data)):
                                    x_batch_list_dev, y_batch_list_dev = zip(*dev_data[start_index_dev: end_index_dev])
                                else:
                                    x_batch_list_dev, y_batch_list_dev = zip(*dev_data[start_index_dev: len(dev_data)])

                                x_batch_dev = np.array(x_batch_list_dev)
                                y_batch_dev = np.array(y_batch_list_dev)
                                start_index_dev += FLAGS.batch_size
                                end_index_dev += FLAGS.batch_size
                                if end_index_dev>len(dev_data):
                                    break
                                try:
                                    step_loss_dev, step_accuracy_dev = dev_step(x_batch_dev, y_batch_dev,writer=dev_summary_writer)
                                    loss_dev += step_loss_dev
                                    accuracy_dev += step_accuracy_dev
                                    count += 1
                                except Exception as ex:
                                    print("dev error!")
                            time_str = datetime.datetime.now().isoformat()
                            #print ("dev {}: step {}, loss {:g}, acc {:g}".format(time_str, current_step,loss_dev / count, accuracy_dev / count))
                            print("\ndev:"+time_str+" step "+str(current_step)+" loss "+str(loss_dev/count)+" accuracy "+str(accuracy_dev/count)+"\n")
                            sys.stdout.flush()
                       
                        if current_step % FLAGS.checkpoint_every == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print ("Saved model checkpoint to {}\n".format(path))
                            sys.stdout.flush()
                    except Exception as err:
                        print(err)
                    
                    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--log_dir', type=str,
                        default='hdfs://localhost:50070/tf_data/local/board',
                        help='Summaries log directory')
    



    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='word_embedding')

    parser.add_argument('--filter_sizes', type=str, default="1,2,3,4",
                        help='filter_sizes')

    parser.add_argument('--num_filters', type=int, default=128,
                        help='number os filters per filter size') 

    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='dropout_keep_prob') 
    
    parser.add_argument('--print_every', type=int, default=100,
                        help='print_every') 
    
    parser.add_argument('--l2_reg_lambda', type=float, default=0.001,
                        help='l2_reg_lambda')

    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Initial learning rate')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch_size') 

    parser.add_argument('--dev_batch_size', type=int, default=20,
                        help='dev_batch_size') 

    parser.add_argument('--evaluate_every', type=int, default=1000,
                        help='evaluate_every') 

    parser.add_argument('--checkpoint_every', type=int, default=1000,
                        help='checkpoint_every') 


    parser.add_argument('--num_classes', type=int, default=10,
                        help='num_classes') 
    
    parser.add_argument('--allow_soft_placement', type=bool, default=True,
                        help='num_classes') 
    
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='num_classes') 

    parser.add_argument('--num_epochs',type=int,default=20,help='epochs')

    FLAGS, unparsed = parser.parse_known_args()
    log_dir = FLAGS.log_dir
    #if tf.gfile.Exists(FLAGS.log_dir):
    #    tf.gfile.DeleteRecursively(FLAGS.log_dir)
    #tf.gfile.MakeDirs(FLAGS.log_dir)
    


    print("----fininshed read args---")
    train()

