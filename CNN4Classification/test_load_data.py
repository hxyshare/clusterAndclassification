#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:37:29 2018

@author: huaxinyu
"""
#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers_fix as data_helpers
from text_cnn_mulconv import TextCNN
import pickle as pkl
import sys
import random
import gensim

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