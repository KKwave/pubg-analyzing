#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

location = r'J:\Kaggle\PUBG_data\\' # 项目地址，最后是 \\

# 获取所有文档名称并存到列表中
def get_file_name():
    filename = []
    os.chdir(location) # 改变当前目录
    os.listdir(location) # 展示目录下文件
    for files in os.listdir('.'): #.显示当前目录的文件并获取目标文档名列表
        if files.startswith('agg_match_stats') and files.endswith('.csv'):
            filename.append(files)
    print(filename)
    return filename

# 获取文档
def get_file(filename):
    location1 = location+filename # 读取文档列表里的文档名
    try:
        return pd.read_csv(location1,sep=',',header=0)
    except Exception as e:
        print('Error:', e)

# 获取多少份文档
def get_num_list(n):
    return list(range(n))

if __name__=='__main__':
    filename = get_file_name()
    dflist = [get_file(filename[i]) for i in get_num_list(2)]  # 获取两份文档并拼接
    meta_data = pd.concat(dflist,ignore_index = True)
    print('数据形状')
    print(meta_data.shape)

## 数据预览
# print('数据前五行')
# print(meta_data.head(5))
# print('数据形状')
# print(meta_data.shape)
# print('数据列项')
# print(meta_data.columns)
# print('数据信息')
# print(meta_data.info)
# print('---------------------------')