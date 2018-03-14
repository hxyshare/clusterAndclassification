#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:13:44 2018

@author: huaxinyu
"""
#tf-idf特征和word2vec的特征
import numpy as np  
from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer    
import math
import os
import jieba
import pandas as pd
from sklearn import metrics

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(vector):
    return math.sqrt(dot_product(vector, vector))

def similarity(v1, v2):
    '''计算余弦相似度

    '''
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)

def write_file(name):
    fw = open('/Users/huaxinyu/codes/'+name,'w')


def get_vocabulary():
    vocabulary = {}
    count = 0
    for i in open('/Users/huaxinyu/codes/wvdict.txt').readlines():
        res = i.strip().split()
        vocabulary[res[0]] = count
        count += 1
    tmp = list(vocabulary)
    
    index2word = tmp
    word2index = {x:i for i ,x in enumerate(index2word)}
    return vocabulary,index2word,word2index

#" ".join(jieba.cut(string2))
def get_word2vec(index2word):
    import gensim
    import random
    model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('/Users/huaxinyu/codes/user_shouwen_1.model')
    W = []
    #get_vocabulary()
    for v in word2index:
        try:
            W.append(np.float32(model[v]))
        except:
            l = []
            for i in range(128):
                l.append(random.uniform(-0.5,0.5))
            W.append(np.float32(np.array(l)))
            print("ecx")
    W = np.array(W)
    return W
    
#得到句向量的均值。
def get_vector(sentence):
        res = np.zeros([128])
        count = 0
        for word in sentence.split():
            #print(word)
            res += W[word2index[word]]
            count += 1
        return res/count

def get_sentence_vec(sens):
    sentences = []
    for sentence in sens:
         sentences.append(get_vector(sentence))
    return np.array(sentences)

def get_word2vec_data(file_path):
    names = [os.path.join(file_path,f) for f in os.listdir(file_path)]
    test_labels = []
    names.remove('/Users/huaxinyu/codes/corpus/.DS_Store')
    #print(len(names))
    corpus = []
    
    maxLineNum = 100
    for i in names:    
        lineNum = 0
        for line in open(i, 'r').readlines():
            if lineNum <50:
                #print(line)
                corpus.append(" ".join(jieba.cut(line.strip())))
            else:   
                test_data.append(" ".join(jieba.cut(line.strip())))
                test_labels.append(i[32:-4])
            #print(i[32:-4])
            lineNum += 1
            if lineNum == maxLineNum:
                break
    print('corpus',get_sentence_vec(corpus).shape)
    #print('test_data',len(get_sentence_vec(test_data)))
    #print('test_label',len(test_labels))
    return get_sentence_vec(corpus),get_sentence_vec(test_data),test_labels
#重新进行，数据的
def get_tfidf_data(file_path):
    names = [ os.path.join(file_path,f) for f in os.listdir(file_path) ]
    #names.remove('/Users/huaxinyu/codes/corpus/.DS_Store')
    #print(len(names))
    corpus = []
    test_data = []
    test_labels = []
    maxLineNum = 100
    for i in names:    
        lineNum = 0
        for line in open(i, 'r').readlines():
            #print(names,len(open(i, 'r').readlines()))
            #print(line)
            if lineNum <50:
                corpus.append(" ".join(jieba.cut("".join(line.split()))))
                #print(" ".join(jieba.cut("".join(line.split()))))
            else:   
                test_data.append(" ".join(jieba.cut(line.strip())))
                test_labels.append(i[29:-4])
            #print(i[32:-4])
            lineNum += 1
            if lineNum == maxLineNum:
                break
    
    print('testdata',len(test_data))
    vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b')  
      
    #print(word)
    analyze = vectorizer.build_analyzer()
    print(analyze("我 在 昆仑 路 杭州 小笼包 店 买 的 炒 米粉 备注 所有 的 辣 都 不放 送来 辣 的 没法 吃"))
    transformer = TfidfTransformer()  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus+test_data)) 
    word = vectorizer.get_feature_names()
    #print(word)
    fixed_weight = tfidf.toarray()  
    #print(len(fixed_weight[0:500]))
    print(fixed_weight.shape)
    vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b')  
    transformer = TfidfTransformer()  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(test_data+corpus))  
    #word = vectorizer.get_feature_names()  
    test_w = tfidf.toarray()  
   # print(len(test_w[500:1000]))
    return fixed_weight[:500],test_w[0:500],test_labels
#直接就是输出了0-1之间的值。
def gen_sim_cos(A,B):
    num = float(np.dot(A,B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim

def gen_sim_euclidean(A,B):
    dist = linalg.norm(A - B)  
    sim = 1.0 / (1.0 + dist) #归一化 
    return sim
    
#选定固定的10个分类点。tf-idf特征相加，词向量特征
#dataset里面就是50 * 10数据
def fixedCent(dataSet, k):
    m = 0
    n = np.shape(dataSet)[1]
    centroids = np.array(np.zeros((k,n)))#
    for j in range(k):
        for i in range(50):
            centroids[j,:] += dataSet[i+m]
        m += 50
        centroids[j,:] /= 50.0
    return centroids
def kMeans(fixed_w,test_w ,l,k,distMeas=gen_sim_cos, createCent=fixedCent):
    #threshold,如果大于阈值的话，就把某个k的标记作为label，否则把最大的标记为
    #就说我我这个样本去和每一个样本进行计算都要进行比阈值的比较
    #不同的阈值的情况下。效果是怎么样的
        
    threshold = 0.8
    lowthreshold = 0.4
    m = np.shape(fixed_w)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    
    centroids = createCent(fixed_w, k)
    #print(len(centroids))
    counter = 0
    
    #print(k)
    while counter <= 4:
        counter += 1
        wrongCase = 0
        labels =[] 
        clusterChanged = False
        for i in range(m):#哪个距离最近 遍历每一个样本
            maxSim = -1; 
            #print('--------------------------')
            maxIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],test_w[i,:])
                #print(distJI)
                if distJI > maxSim:
                    maxSim = distJI; 
                    maxIndex = j
            #真实标记的值不等于求出来的最大值。
            #print(names[maxIndex][29:-4])
            #print(l[i])
            if l[i] != names[maxIndex][29:-4]: 
                    #print(str('/Users/huaxinyu/codes/traindata/'+l[i]+'.txt'))
                    tmp_e = names2index[('/Users/huaxinyu/codes/corpus/'+l[i]+'.txt')]
                    
                    realsim = distMeas(centroids[tmp_e,:],fixed_w[i,:])
                    wrongCase += 1
                    #真实的label，测试数据，与哪个簇距离最近，与最近簇最大的相似度，与自己的真实label的相似的度 
                    #print(l[i],test_data[i],names[maxIndex][32:-4],maxSim,realsim)
            #这个只是简单的把中心点聚合，label可能有错的情况。
            labels.append(names2index[names[maxIndex]])
            #print(len(labels))
            if maxSim >threshold and l[i] == names[maxIndex][29:-4]:
                clusterAssment[i,:] = maxIndex,maxSim
            else:
                clusterAssment[i,:] = maxIndex,maxSim
                
        print('wrong',wrongCase)
        
        
        for cent in range(k):#，把新的样本的点。重新计算中心点，
            ptsInClust = test_w[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            #print(len(ptsInClust[1]))
            #print('o=pstsinclust',ptsInClust.shape)
            #c =  np.array(np.array(list(fixed_w[cent])*50).reshape(50,-1))
            c =  np.array([fixed_w[cent]])
            #print(c.shape)
            b = np.concatenate((c,np.array(ptsInClust)),axis=0)
            #print(b.shape)
            t = np.array(np.mean(b,axis=0))
            #print(t)
            #print(gen_sim_cos(t,c ))
            centroids[cent,:] = t
            
    return labels

def Unlabel_kMeans(fixed_w,test_w ,l,k,distMeas=gen_sim_cos, createCent=fixedCent):
    #threshold,如果大于阈值的话，就把某个k的标记作为label，否则把最大的标记为
    #就说我我这个样本去和每一个样本进行计算都要进行比阈值的比较
    #不同的阈值的情况下。效果是怎么样的
        
    threshold = 0.8
    lowthreshold = 0.4
    m = np.shape(fixed_w)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    
    centroids = createCent(fixed_w, k)
    #print(len(centroids))
    counter = 0
    
    #print(k)
    while counter <= 4:
        counter += 1
        wrongCase = 0
        labels =[] 
        clusterChanged = False
        for i in range(m):#哪个距离最近 遍历每一个样本
            maxSim = -1; 
            #print('--------------------------')
            maxIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],test_w[i,:])
                #print(distJI)
                if distJI > maxSim:
                    maxSim = distJI; 
                    maxIndex = j
          
            
            #print(len(labels))
            if maxSim >threshold :
                clusterAssment[i,:] = maxIndex,maxSim
            #如果最大的相似度还是小于lowthreshold，说明他是来自于其他的新的类别
            #我也不知道有多少新的类别啊，。。。
        
            elif maxSim < lowthreshold:
                clusterAssment[i,:] = maxIndex,maxSim
                
        
        
        for cent in range(k):#，把新的样本的点。重新计算中心点，
            ptsInClust = test_w[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            #print(len(ptsInClust[1]))
            #print('o=pstsinclust',ptsInClust.shape)
            #c =  np.array(np.array(list(fixed_w[cent])*50).reshape(50,-1))
            c =  np.array([fixed_w[cent]])
            #print(c.shape)
            b = np.concatenate((c,np.array(ptsInClust)),axis=0)
            #print(b.shape)
            t = np.array(np.mean(b,axis=0))
            #print(t)
            #print(gen_sim_cos(t,c ))
            centroids[cent,:] = t
            
    return labels
vocabulary,index2word,word2index = get_vocabulary()

print('vocabulary',len(index2word))
#W = get_word2vec(index2word)
#print("W",len(W))
#corpus = []
col_names = [f[:-4] for f in os.listdir('/Users/huaxinyu/codes/corpus')] 
#print(col_names)
#col_names.remove('.DS_S')
test_data = []
fixed_w,tets_w,label = get_tfidf_data('/Users/huaxinyu/codes/corpus')
#fixed_w,tets_w,label = get_word2vec_data('/Users/huaxinyu/codes/traindata')
#print('corpus',len(corpus))
#print(label)
labelindex = {i:x for x,i in enumerate(label)}
#print(len(w))
#print(len(c))
#g = fixedCent(w,10)
f_p = '/Users/huaxinyu/codes/corpus'
names = [ os.path.join(f_p,f) for f in os.listdir(f_p) ]
#names.remove('/Users/huaxinyu/codes/corpus/.DS_Store')
names2index = {i:x for x,i in enumerate(names)}
print(names2index)
#print(index2names)
#print(index2names[1][32:])
#print(len(w))
#print(len(g[1]))
pred = kMeans(fixed_w,tets_w,label,10)
#print(pred)
real_label = []
for i in label:
    real_label.append(names2index[str('/Users/huaxinyu/codes/corpus/'+i+'.txt')])
#print(len(real_label))
#print(real_label)
print(metrics.classification_report(real_label, pred,target_names=col_names)) 
