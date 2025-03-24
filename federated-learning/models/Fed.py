#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    # .deepcopy() 用于创建一个对象的深度副本（deep copy），对象的值和原始对象相同，但是在内存中具有不同的地址，以避免对原始对象的修改造成的影响。
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        # 对累积的值进行平均，将其除以参数列表 w 的长度。torch.div() 函数用于执行元素级除法。
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

"""""
#举个例子
student1 = {
    'name': 'John',
    'age': 18,
    'grade': '12th',
    'school': 'ABC High School'
}

student2 = {
    'name': 'Jane',
    'age': 17,
    'grade': '11th',
    'school': 'XYZ High School'
}
student3 = {
    'name': 'hidisan',
    'age': 23,
    'grade': '14th',
    'school': 'NB High School'
}

w = [student1, student2, student3]

#那w_avg一开始等于student1,
w_avg = copy.deepcopy(w[0])

#经过遍历累加得到
w_avg = {'name': 'JohnJanehidisan', 
         'age': 58, 
         'grade': '12th11th14th', 
         'school': 'ABC High SchoolXYZ High SchoolNB High School'}
#然后经过torch.div,执行元素级除法
"""""