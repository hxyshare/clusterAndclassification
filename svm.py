#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:06:36 2018

@author: huaxinyu
"""

from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer    
import math
import os
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier 
from sklearn.grid_search import GridSearchCV
import  numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import HashingVectorizer 

class GetData:

    def get_data(self ,file_path):
        names = [f[:-4] for f in os.listdir('/Users/huaxinyu/codes/traindata')] 
        names.remove('.DS_S')
        index2names = [ os.path.join(file_path,f) for f in os.listdir(file_path) ]
        y = []
        index2names.remove('/Users/huaxinyu/codes/traindata/.DS_Store')
        names2index =  {i:x for x,i in enumerate(index2names)} 
        print(names2index)
        #print(len(names))
        x = []
        
        maxLineNum = 100
        for i in index2names:    
            lineNum = 0
            for line in open(i, 'r').readlines():
                    #print(i)
                    x.append(" ".join(jieba.cut(line.strip())))
                    y.append(names2index[i])
                    
                    lineNum += 1
                    if lineNum == maxLineNum:
                        break
    #    print(y)
    #    vect = CountVectorizer()  
    #    X_train= vect.fit_transform(x)
    #    #get word tf-idf  
    #    tfidf_transformer = TfidfTransformer()  
    #    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    #    x_tfidf = X_train_tfidf.toarray()
    #    # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = np.array(x)
        y = np.array(y)
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        print(len(x),len(y))
        
        dev_sample_index = -1 * int(0.2* float(len(y_shuffled)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.index2names = index2names
        self.names2index = names2index
        self.names = names
       
        
def test_tfidf():
    #get vector  
    vect = CountVectorizer()  
    X_train= vect.fit_transform(twenty_train.x_train)
    #get word tf-idf  
    tfidf_transformer = TfidfTransformer()  
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)  
    #model train  
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.y_train)  
    docs_new = ['骑手 送的 太慢', '饭 里面 有 虫子']  
    X_new = vect.transform(docs_new)  
    X_new_tfidf = tfidf_transformer.transform(X_new)  
    #predict  
    predicted = clf.predict(X_new_tfidf)   
    for doc, category in zip(docs_new, predicted):   
        print('%r => %s' % (doc, twenty_train.index2names[category]))   


def testPipline():  
     
#    #1. MultinomialNB  
#    text_clf = Pipeline([('vect', CountVectorizer()),   
#                ('tfidf', TfidfTransformer()),   
#                ('clf', MultinomialNB()),   
#                ])  
#    text_clf.fit(data.x_train, data.y_train)  
#      
#    docs_test = data.x_dev
#    nb_predicted = text_clf.predict(docs_test)  
#      
#    accuracy=np.mean(nb_predicted == data.y_dev)  
#    #print accuracy   
#    print ("The accuracy of twenty_test is %s" %accuracy)  
#      
#    print(metrics.classification_report(data.y_dev, nb_predicted,target_names=data.names))  
#      
#    #2. KNN  
#    text_clf = Pipeline([('vect', CountVectorizer()),   
#                ('tfidf', TfidfTransformer()),   
#                ('clf', KNeighborsClassifier()),   
#                ])  
#    text_clf.fit(data.x_train, data.y_train)  
#      
#    docs_test = data.x_dev
#    knn_predicted = text_clf.predict(docs_test) 
#      
#    accuracy=np.mean(knn_predicted == data.y_dev)  
#    #print accuracy   
#    print ("The accuracy of twenty_test is %s" %accuracy)  
#      
#    print(metrics.classification_report(data.y_dev, knn_predicted,target_names=data.names))  
#      
    #3. SVM  
    text_clf = Pipeline([('vect', CountVectorizer()),  
                     ('tfidf', TfidfTransformer()),  
                     ('clf', SGDClassifier(max_iter=1000,loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])  
      
    text_clf.fit(data.x_train, data.y_train)  
      
    docs_test = data.x_dev
    svm_predicted = text_clf.predict(docs_test)
    print(data.x_dev[svm_predicted!=data.y_dev],data.y_dev[svm_predicted!=data.y_dev],svm_predicted[svm_predicted!=data.y_dev])
    import pandas as pd
    #print(data.names)
    i2n = {i:x for i,x in enumerate(data.names)}
    s1 = pd.Series(data.x_dev[svm_predicted!=data.y_dev])
    s2 = pd.Series([i2n[i]for i in data.y_dev[svm_predicted!=data.y_dev]])
    s3 = pd.Series([i2n[i]for i in svm_predicted[svm_predicted!=data.y_dev]])
    res = pd.DataFrame({'a':s1,'b':s2,'c':s3})
    #print(res)
    accuracy=np.mean(svm_predicted == data.y_dev) 
    #print(text_clf.decision_function(data.x_dev[svm_predicted!=data.y_dev]))
    #print(np.max(text_clf.decision_function(data.x_dev[svm_predicted!=data.y_dev]),axis = 1))
    t = text_clf.decision_function(data.x_dev[svm_predicted!=data.y_dev])
    b = [a[np.argpartition(a,-2)[-2:]][::-1] for a in t]
    #c = [(b[vv] for vv in i) for i in b]
    
    print(b)
    print(c)
    #print((a[np.argpartition(a,-2)[-2:]][::-1]))
    #print accurcy   
    print ("The accuracy of twenty_test is %s" %accuracy)  
    res.to_csv('/Users/huaxinyu/codes/res.txt',header = False,index = False,encoding ='utf-8')
    print(metrics.classification_report(data.y_dev, svm_predicted,target_names=data.names))
  
#  
#    #4. 少量特征  
# 
#    text_clf = Pipeline([('vect', HashingVectorizer(stop_words = 'english',non_negative = True,    
#                               n_features = 10000)),  
#                     ('tfidf', TfidfTransformer()),  
#                     ('clf', SGDClassifier(max_iter=1000,loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])  
#      
#    text_clf.fit(data.x_train, data.y_train)  
#      
#    docs_test = data.x_dev
#    hsah_predicted = text_clf.predict(docs_test) 
#    #print accuracy   
#    print ("The accuracy of twenty_test is %s" %accuracy)  
#     
#    print(metrics.classification_report(data.y_dev, hsah_predicted,target_names=data.names))  



    #print(metrics.confusion_matrix(data.y_dev, nb_predicted) )
    #print(metrics.confusion_matrix(data.y_dev, knn_predicted) )
    print(metrics.confusion_matrix(data.y_dev, svm_predicted) )
    #print(metrics.confusion_matrix(data.y_dev, hsah_predicted) )
    





#GridSearchCV 搜索最优参数  
def testGridSearch():  
    print( '*************************\nPipeline+GridSearch+CV\n*************************'  )
    text_clf = Pipeline([('vect', CountVectorizer()),  
                     ('tfidf', TfidfTransformer()),  
                     ('clf', SGDClassifier(max_iter=1000)),])  
      
    parameters = {    
      'vect__ngram_range': [(1, 1), (1, 2)],  
      'vect__max_df': (0.5, 0.75),    
      'vect__max_features': (None, 5000, 10000),    
      'tfidf__use_idf': (True, False),    
    #  'tfidf__norm': ('l1', 'l2'),    
       'clf__alpha': (0.00001, 0.000001),  
       #'max_iter' :(1000,2000)
    #  'clf__penalty': ('l2', 'elasticnet'),    
    #   'clf__n_iter': (10, 50),    
    }    
    #GridSearch 寻找最优参数的过程  
    flag = 0
    if (flag!=0):  
        grid_search = GridSearchCV(text_clf,parameters,n_jobs = 1,verbose=1)  
        grid_search.fit(twenty_train.data, twenty_train.target)     
        print("Best score: %0.3f" % grid_search.best_score_)   
        best_parameters = dict();   
        best_parameters = grid_search.best_estimator_.get_params()    
        print("Out the best parameters");    
        for param_name in sorted(parameters.keys()):   
            print("\t%s: %r" % (param_name, best_parameters[param_name]));    
      
    #找到最优参数后，利用最优参数训练模型  
    text_clf.set_params(clf__alpha = 1e-05,     
                    tfidf__use_idf = True,    
                    vect__max_df = 0.5,    
                    vect__max_features = None);    
    text_clf.fit(twenty_train.data, twenty_train.target)  
    #预测  
    pred = text_clf.predict(twenty_test.data)  
    #输出结果  
    accuracy=np.mean(pred == twenty_test.target)  
    #print accuracy   
    print ("The accuracy of twenty_test is %s" %accuracy)  
     
    print(metrics.classification_report(twenty_test.target, pred,target_names=twenty_test.target_names))  
    array = metrics.confusion_matrix(twenty_test.target, pred)  
    print (array  )   


#test()
#testGridSearch()
#testPipline()
# Split train/test set
data = GetData()
data.get_data('/Users/huaxinyu/codes/traindata')
#test_tfidf()
testPipline()
