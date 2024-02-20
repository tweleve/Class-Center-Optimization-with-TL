# -*- coding: UTF-8 -*-
import os
import random
import numpy
import shutil


def creat_txt(root_path, txt_name, num):
    newRoot_path = root_path + txt_name + '/'
    files = os.listdir(newRoot_path)
    # maxId = len(files)
    labelId = num
    cur_lines = []
    for file in files:
        newfiles = os.listdir(newRoot_path + file + '/')
        stop_flag = (len(newfiles)) / 2  # 取一半的数据作为数据集，参与train，test
        flag = 0  # 选取的图片数
        for i in newfiles:

            cur_dir = txt_name + '/' + file + '/' + i
            cur_id = labelId
            cur_lines.append(cur_dir + ' ' + str(cur_id))
            flag += 1
            if flag >= stop_flag:
                break
        labelId += 1
    file_name = root_path + txt_name + '_50.txt'
    print(file_name)
    with open(file_name, 'w') as f:
        for cur_line in cur_lines:
            f.write(cur_line + '\n')


def get_dict_path():
    path_dict = {  # 可用于设计数据集重命名循环
        'SIRI-WHU-Google-12': './datasets/SIRI-WHU-Google-12/',
        'WHU-RS-19': './datasets/WHU-RS-19/',
        'UCMD-21': './datasets/UCMD-21/',
        'CLRS-25': './datasets/CLRS-25/',
        'AID-30': './datasets/AID-30/',
        'OPTIMAL-31': './datasets/OPTIMAL-31/',
        'RSI-CB256-35': './datasets/RSI-CB256-35/',
        'PatternNet-38': './datasets/PatternNet-38/',
        'NWPU-45': './datasets/NWPU-45/'
    }
    return path_dict


if __name__ == '__main__':
    # path_dict = get_dict_path()
    # path_dict = {'AID-30': './datasets/AID-30/', }
    # path_dict = {'UCMD-21': './datasets/UCMD-21/', }
    # path_dict = {'PatternNet-38': './datasets/PatternNet-38/', }
    path_dict = {'OPTIMAL-31': './datasets/OPTIMAL-31/', }
    for key in path_dict:
        root_path = path_dict[key]
        creat_txt(root_path, 'train', 0)  # train下标从0开始
        creat_txt(root_path, 'test', 16)  # test下标从N/2开始
