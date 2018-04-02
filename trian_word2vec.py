#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:24:01 2018

@author: huaxinyu
"""


from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec 
sentences = LineSentence('/Users/huaxinyu/codes/preprocess.txt')

model = Word2Vec(sentences, size=128, window=8, min_count=4, workers=4) 
#model.save('/Users/huaxinyu/Downloads/user_shouwen.model')
model.wv.save_word2vec_format("/Users/huaxinyu/Downloads/user_shouwen_1.model",fvocab='/Users/huaxinyu/Downloads/wvdict.txt')

import gensim
model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('/Users/huaxinyu/codes/user_shouwen_1.model')
    
import pandas as pd
print(pd.DataFrame(model.most_similar('外卖')))
