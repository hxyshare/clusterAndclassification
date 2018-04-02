# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@author: huaxinyu
"""
import numpy as np
import os
import re
import jieba
#from gensim.models import Word2Vec
from  gensim.models.word2vec import LineSentence
result = {}
for line in open('/Users/huaxinyu/Downloads/外卖聊天日志.txt'):
    splitdata = line.strip().split()
    if splitdata[1] not in result:
        tmp = []
        result[splitdata[1]] = splitdata[2]
        
        #result[splitdata[1]] = tmp.extend(splitdata[2])
        #print('id'+splitdata[1])
        #print(result[splitdata[1]])
    else:
        result[splitdata[1]] += splitdata[2]
        #print('id'+splitdata[1])
        #print(result[splitdata[1]])
print(len(result))

wfile = open('/Users/huaxinyu/Downloads/waimailog.txt','w')
for i in result:
    wfile.write(i+'\t'+result[i]+'\n')
    print (i+'\t'+result[i]+'\n')
print('done')
wfile.close()

fenci  = {}
for line in  open('/Users/huaxinyu/Downloads/waimailog.txt'):
    s = line.split('\t')
    seg_list = jieba.cut(s[1])
    fenci[s[0]] = seg_list
    #print(" ".join(seg_list))
    
wfenci = open('/Users/huaxinyu/Downloads/waimailog_fenci.txt','w')
for i in fenci:
    wfenci.write(" ".join(fenci[i])+'\t')   
print("fenci done")
wfenci.close()


from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec 
sentences = LineSentence('/Users/huaxinyu/codes/preprocess.txt')

model = Word2Vec(sentences, size=128, window=8, min_count=5, workers=4) 
#model.save('/Users/huaxinyu/Downloads/user_shouwen.model')
model.wv.save_word2vec_format("/Users/huaxinyu/Downloads/user_shouwen_1.model",fvocab='/Users/huaxinyu/Downloads/wvdict.txt')

import gensim
model = gensim.models.Word2Vec.load('/Users/huaxinyu/codes/user_shouwen_1.model')
import pandas as pd
print(pd.DataFrame(model.most_similar('外卖')))


from sklearn.manifold import TSNE

def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()
 
 
words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)
