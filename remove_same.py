#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:10:42 2018

@author: huaxinyu
"""

#-*- coding: utf-8 -*-
import codecs
def removRep(char_list):
    list1=['']
    list2=['']
    del1=[]
    flag=['']
    #print(char_list)
    i=0
    while(i<len(char_list)):
        if (char_list[i]==list1[0]):
            if (list2==['']):
                list2[0]=char_list[i]
            else:
                if (list1==list2):
                    t=len(list1)
                    m=0
                    while(m<t and i!=len(char_list)-1):
                        del1.append( i-m-1)
                        #print(list1,list2,i,del1)
                        
                        m=m+1
                    list2=['']
                    list2[0]=char_list[i]
                else:
                    list1=['']
                    list2=['']
                    flag=['']
                    list1[0]=char_list[i]
                    flag[0]=i       
        else:
            if (list1==list2)and(list1!=[''])and(list2!=['']):
                if len(list1)>=2:
                    t=len(list1)
                    m=0
                    while(m<t):
                        del1.append( i-m-1)
                       
                        m=m+1  
                    list1=['']
                    list2=['']
                    list1[0]=char_list[i]
                    flag[0]=i
            else:
                if(list2==['']):
                    if(list1==['']):
                        list1[0]=char_list[i]
                        flag[0]=i
                    else:
                       list1.append(char_list[i])
                       flag.append(i)
                else:
                    list2.append(char_list[i])
        i=i+1
        if(i==len(char_list)):
           if(list1==list2):
                    t=len(list1)
#                    m=0
#                    while(m<t):
#                        del1.append( i-m-1)
#                        #print(i,del1,flag)
#                        m=m+1
                    m=0
                    while(m<t):
                        del1.append(flag[m])
                        #print(i,del1)
                        m=m+1  
                                   
   #print(list1,list2)
    a=sorted(del1)
    #print(a)
    t=len(a)-1
    while (t>=0):
        #print(char_list[a[t]])
        del char_list[a[t]]
        t=t-1
    str1 = "".join(char_list[::-1])  
    str3 = "".join(list1)
    str2=str1.strip() #删除两边空格 
    if len(str1)!= 0:
        print(str2)
    else:
        print(str3)
#   f1.close()
        
        
inputfile = '/Users/huaxinyu/codes/same2.txt' #评论文件
#outputfile = 'H_KJ300F-JAC2101W_process_2.txt' #评论处理后保存路径
f = codecs.open(inputfile ,'r','utf-8')
#f1=codecs.open(outputfile,'w','utf-8')
fileList = f.readlines()
f.close()
for A_string in fileList: 
    temp1= A_string.strip('\n')       #去掉每行最后的换行符'\n' 
    temp2 = temp1.lstrip('\ufeff') 
    temp3= temp2.strip('\r')
    char_list=list( temp3)[::-1]
    removRep(char_list)
   