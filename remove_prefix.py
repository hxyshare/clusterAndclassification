#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:15:52 2018

@author: huaxinyu
"""

import re
s = '哈哈哈哈哈哈哈哈哈你们送了吗'
print(s.replace('.*?\d+?\\t ',''))