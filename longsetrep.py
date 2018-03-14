#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:29:29 2018

@author: huaxinyu
"""

from collections import Counter  
  
  
def slice_window(one_str,w=1):  
    ''''' 
    滑窗函数 
    '''  
    res_list=[]  
    for i in range(0,len(one_str)-w+1):  
        res_list.append(one_str[i:i+w])  
    return res_list  
  
  
def main_func(one_str):  
    ''''' 
    主函数 
    '''  
    all_sub=[]  
    for i in range(1,len(one_str)):  
        all_sub+=slice_window(one_str,i)  
    res_dict={}  
    #print Counter(all_sub)  
    threshold=Counter(all_sub).most_common(1)[0][1] 
    print(threshold)
    slice_w=Counter(all_sub).most_common(1)[0][0]  
    for one in all_sub:  
        if one in res_dict:  
            res_dict[one]+=1  
        else:  
            res_dict[one]=1  
    sorted_list=sorted(res_dict.items(), key=lambda e:e[1], reverse=True)  
    tmp_list=[one for one in sorted_list if one[1]>=threshold]
    
    print(tmp_list)
    #python3.0版本的排序，sorted函数的使用。
    sorted_list=sorted(tmp_list, key=lambda x : len(x[0]), reverse=True) 
    a,n = sorted_list[0]
    #print(a,n)
    #print tmp_list  
    print (sorted_list[0][0])
    print(one_str.replace(a,'',n-1))
  
  
#他会先
if __name__ == '__main__':  
    one_str='真的非常非常非常非常非常非常非常非常喜欢你'  
    two_str='为什么为什么安装费这么贵，毫无道理'  
    three_str='bbbbbbb'  
    www = '真的很好很好很好很好用'
    main_func(one_str)  
    main_func(two_str)  
    main_func(www)  