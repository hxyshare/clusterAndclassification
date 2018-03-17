#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:16:24 2018

@author: huaxinyu
"""

import jieba
import os
filedir = '../corpus'
filefenci = '../corpus_fenci'

for i in os.listdir(filedir):
    print (i)
    res = []
    for  j in open('../corpus/' + i, 'r').readlines():
        res.append(jieba.cut(j))
    
        
    wfenci = open('../corpus_fenci/' + i,'w')
    for i in res:
        wfenci.write(" ".join(i))   
    print("fenci done")
    wfenci.close()