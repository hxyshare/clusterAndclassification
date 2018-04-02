#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:55:21 2018

@author: huaxinyu
"""

import numpy as np
import os
import re
import jieba
wf = open('/Users/huaxinyu/Downloads/preprocess.txt','w')
count = 0
unique = {} 
for line in open('/Users/huaxinyu/Downloads/外卖聊天日志.txt'):
    splitdata = line.strip().split()
    if splitdata[3] == '1' and len(splitdata[2])>=5 and not re.findall(r"1\d{8,16}",splitdata[2]) \
    and '<img src' not in splitdata[2] and '&#x' not in splitdata[2] and "谢谢" not in splitdata[2] \
    and "好的" not in splitdata[2] and not re.findall(r"\d{3}.\d{4}.\d{4}",splitdata[2])\
    and len(splitdata[2]) != 0 and not re.findall(r"(\[.*?\])+",splitdata[2]):
        string1 = re.sub("，", "", splitdata[2])
        string2 = re.sub("。","",string1)
        string3 = " ".join(jieba.cut(string2))
        if splitdata[1] not in unique:
            wf.write(string3+'\n')
            unique[splitdata[1]] = 1
            count += 1
            #print(splitdata[1] + string3)
        

wf.close()
print(count)
import jieba
wf = open('/Users/huaxinyu/codes/test_fenci.txt','w')
count = 0
unique = {} 
for line in open('/Users/huaxinyu/codes/test.txt'):
    splitdata = line.strip().split()
    string3 = " ".join(jieba.cut(splitdata[0]))
    wf.write(string3+'\n')
    #print(splitdata[0])