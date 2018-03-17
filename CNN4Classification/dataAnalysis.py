#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:11:53 2018

@author: huaxinyu
"""


import os
import sys 
#reload(sys) 
#ys.setdefaultencoding('utf8') 
word_dict = {}
filepath ='../testdata/' 
sentencecount = 0
count = 0
files = os.listdir(filepath)
for f in files:
      if f != '.DS_Store':
          print(f)
          file_path = os.path.join(filepath,f)
          fileLineLen = len(open(file_path).readlines())
          sentencecount +=fileLineLen
          print (file_path)
          for sentence in open(file_path).readlines():
            x = len(sentence.split(" "))
            count += x
           # print (clean_seg)
            if x not in word_dict:
                word_dict[x] = 1  
            else:  
                word_dict[x] += 1  
with open("./wordCount.txt",'w') as wf2:
  orderList=list(word_dict.values())  
  orderList.sort(reverse=True)  
# print orderList  
  for i in range(len(orderList)):  
    for key in word_dict:  
        if word_dict[key]==orderList[i]:  
            wf2.write(str(key)+' '+str(word_dict[key])+'\n')  
            word_dict[key]=0  
print (count)
print (sentencecount)


import numpy as np
import matplotlib.pyplot as plt

word_dict = {}
f = open('./wordCount.txt')
for i in f.readlines(): 
        res = i.strip().split(' ')
        word_dict[int(res[0])] = int(res[1])
word_dict = sorted(word_dict.items(), key=lambda d: d[0],reverse=False) 
plt.figure(1)
plt.subplot(211)
#print(np.array(word_dict)[:,0])
x = [x[0] for x in word_dict] 
y = [x[1] for x in word_dict] 
plt.axis([0, 80, 0, 15000])
plt.bar(x,y,color="green")  
plt.show()

