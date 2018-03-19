#! /usr/bin/env python
# coding=utf-8
import predict1
import jieba
import sys
import os
import pickle as pkl

############################
#模型加载部分
############################
#最大句子长度500
sequence_lenth = 40

def rm_stopword(text):
    text_list = text.split(' ')
    rm_result = [word for word in text_list if word not in stopword_dict and word != '']
    return ' '.join(rm_result)

#模型路径
cnn_path ='./runs/1521428058/checkpoints/model-5000'

#word2vec模型路径
word2vec_path = './meituanw2v.model'
#需要三个文件 wvm12002.model wvm12002.model.syn0.npy  wvm12002.model.syn1neg.npy
print( "word2vec loaded")

# 加载类别信息  cate.txt
classify = {}
cateIdName = {}
cate = open('./cate.txt', 'r')
for line2 in cate:
    line2_list = line2.strip().split('\t')
    classify[line2_list[1]] = line2_list[2]
    cateIdName[line2_list[2]] = line2_list[0]
print ('加载标签完成\n')
print(classify)
print(cateIdName)
for i in cateIdName:
    print(i)
l = 10  # 加载标签
labels = []
for j in range(l):
    if j % 10 == 0:
        labels.append(1)
    else:
        labels.append(0)
print(labels)
# 加载词表

with open('./data_no_sw.pkl', 'rb') as f:
         loaded_data = pkl.load(f)
         train_data, dev_data, vocabulary ,vocabulary_inv = \
             tuple(loaded_data[k] for k in
                   ['train_data', 'dev_data', 'vocabulary', 'vocabulary_inv'])# 加载词向量矩阵
print ('cibiaojiazaiwanc')
W = predict1.load_wv_model(word2vec_path, vocabulary_inv)
# 加载cnn模型
print (len(vocabulary_inv))
print ("W's shape is",W.shape)
new_vocabulary_length =  len(vocabulary_inv)
cnn_model = predict1.load_cnn(sequence_lenth,W, cnn_path, new_vocabulary_length)
print ("cnn model loaded")
#加载停用词

#file_stopwords = open('../data/stopword.txt')
#stopword_dict = {}
#for line in file_stopwords:
#	 stopword_dict.setdefault(line.strip(), '')
#print ('加载停用词词典完成\n')
#file_stopwords.close()

############################
#parent_path = './testcorpus/'
#parents = os.listdir(parent_path)
#for file1 in parents:
	#以下为测试一个文件的每条content的例子
	#child = os.path.join(parent_path,file1)
f = open('./test_data.txt', 'r')
iline=1

#定义了结果输出文件
resultfile=open('./cls_result.txt','w')
#scores= predict1.predict(sentence, sequence_lenth, vocabulary_invmap, cnn_model, labels)
#score = list(scores[0][0])
#sort_scores = sorted(score, reverse=True)        
#s_top = cateIdName[str(classify[str(score.index(sort_scores[0]))])] 
#score_top = sort_scores[0]
#print s_top,score_top
#setnt = "sdfsd"
count = 0
for line in f:
    iline+=1
    content = line.strip(' ')
    if len(content) > 4:
	# 分词
	# print datetime.datetime.now().isoformat()+'cnn_start'
	# print datetime.datetime.now().isoformat() + 'seg_word'
	#content = rm_stopword(sentence)
	# 判断文本类别，返回文本在各类别上的概率值
     scores = predict1.predict(content, sequence_lenth, vocabulary, cnn_model, labels)
    
     score = list(scores[0][0])
     sort_scores = sorted(score, reverse=True)
#     if(score.index(sort_scores[0])!=1):
#         count+=1
	# 取第一个类别
	#print sort_scores[0]
    s_top = cateIdName[str(classify[str(score.index(sort_scores[0]))])]
    score_top = sort_scores[0]
    resultfile.write(classify[str(score.index(sort_scores[0]) )]+ '\t'+content)
    
    #resultfile.write(str(score.index(sort_scores[0])))
    #resultfile.write(s_top+'\n')
