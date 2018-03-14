# coding=utf-8    
"""  
Created on Thu Mar  8 10:13:44 2018

@author: huaxinyu
"""    

import numpy as np  
from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer    

def get_vector(sentence):
        res = np.zeros([128])
        count = 0
        for word in sentence.split():
            #print(word)
            res += W[word2index[word]]
            count += 1
        return res/count

if __name__ == "__main__":  
      
      
    corpus = []  
    
    #读取语料 一行预料为一个文档  
    for line in open('/Users/huaxinyu/codes/test_fenci.txt', 'r').readlines():  
            corpus.append(line.strip())
    
    #读取词典
    vocabulary = {}
    count = 0
    for i in open('/Users/huaxinyu/codes/wvdict.txt').readlines():
        res = i.strip().split()
        vocabulary[res[0]] = count
        count += 1
    index2word = list(vocabulary)
    word2index = {x:i for i ,x in enumerate(index2word)}
    print(len(index2word))
    print(len(word2index))
 
    
    vectorizer = CountVectorizer(vocabulary=vocabulary)  
  
    transformer = TfidfTransformer()  
  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
    
    
    word = vectorizer.get_feature_names()  
    weight = tfidf.toarray()  
    print(len(    ))
#    #打印特征向量文本内容  

    word2index = {x:i for i ,x in enumerate(word)}
    print(len(word2index))
    print(word2index['外卖'])
    
    #print ('Features length: ' + str(len(word))  )
    #写入词表
#    resName = "/Users/huaxinyu/Downloads/dict_shouwen.txt"  
#    result = codecs.open(resName, 'w', 'utf-8')  
#    for j in range(len(word)):  
#        result.write(word[j] + '\n')  
        #print(word[j])
#    result.write('\r\n\r\n')  
#  
#    for i in range(len(weight)):  
#       #print( u"-------这里输出第",i,u"类文本的词语tf-idf权重------" )   
#        for j in range(len(word)):  
#            #print(weight[i][j]+'\n') 
#            result.write(str(weight[i][j]) + ' ')  
#        result.write('\r\n\r\n')  
#  
#    result.close()  
    import gensim
    import random
    #model = gensim.models.Word2Vec.load('/Users/huaxinyu/Downloads/user_shouwen_1.model')
    model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('/Users/huaxinyu/codes/user_shouwen_1.model')
    #这个东西没用啊，相当于我在初始化的时候进行了初始化权重。
    W = []
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
    print(len(W))
    
    
    sentences = []
    for sentence in corpus[0:2000]:
         sentences.append(get_vector(sentence))
    
    print(len(sentences))
    
    print ('Start Kmeans:'  )
    from sklearn.cluster import KMeans  
    clf = KMeans(n_clusters=2)  
    #s = clf.fit(weight)  
    
    
    s = clf.fit(sentences)
    
    print(s)
    print(clf.cluster_centers_)  
      
    print(clf.labels_)  
    i = 1 
    
    wf = open('/Users/huaxinyu/codes/result.txt','w')
    while i <= len(clf.labels_):  
        #print(clf.labels_[i-1] ,corpus[i] )
        wf.write(str(clf.labels_[i-1])+ "\t" +corpus[i-1]+ "\n")
        i = i + 1  
  
    print(clf.inertia_)  
    
    
      
        
#    from sklearn.cluster import DBSCAN
#    y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(weight[0:1000])
#    print(y_pred)
#m = 100000
#min_i = 0
#for i in range(1,300,10):
#    clf = KMeans(n_clusters=i)
#    s = clf.fit(weight[0:500])
#    if clf.inertia_ < m:
#        m = clf.inertia_ 
#        min_i = i
#    print( i , clf.inertia_)
#print(i,m)
#    
    
    