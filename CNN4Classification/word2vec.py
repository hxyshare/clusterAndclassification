# coding=utf-8
import numpy as np
import os
import re
import gensim
import logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec 
sentences = LineSentence('/Users/huaxinyu/codes/testdata/data')

model = Word2Vec(sentences, size=128, window=8, min_count=5, workers=4) 
model.save('./meituanw2v.model')
#model.wv.save_word2vec_format("./meituancomment.model",fvocab='./wvdict.txt')
#gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format
#model.save
model = gensim.models.Word2Vec.load('./meituanw2v.model')
import pandas as pd
print(pd.DataFrame(model.most_similar('外卖')))
print(pd.DataFrame(model['外卖']))