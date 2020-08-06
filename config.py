#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/1 6:39 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : config.py

import os

def get_keyword_str(file):
    key_str = u''
    with open(file, 'r', encoding='utf8') as fin:
        # fin.readline()
        # lines = fin.readlines()
        # if len(lines) > 0:
        #     key_str += lines[0]
        for line in fin:
            key_str += line.replace('\n', '')
        fin.close()
    return key_str


class cnn_ctc_conf(object):
    def __init__(self, gpu=-1):
        self.gpu = gpu
        if self.gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.gpu_frac_mem = 0.8
        self.batch_size = 1
        self.max_epoch_size = 20
        self.train_image_path = './data/images'
        self.train_data_file = './data/train.txt'
        self.test_image_path = './data/images'
        self.test_data_file = './data/test.txt'
        self.image_type = 'jpg'
        self.image_max_width = 280
        self.image_max_height = 32
        self.max_label_length = 18

        # for train
        self.learning_rate_init = 0.01 # 0.0005
        self.learning_rate_decay = 0.6 # decay for learning rate
        self.optimizer = 'adam' # deault is adam
        self.learning_type = 'exponential'  # ['exponential','fixed','polynomial']
        self.patience = 3
        self.dropout = 1.0

        self.model_path = './output'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.trained_model_file_name = 'densenet_ctc.h5'
        self.init_epoch = 0
        self.logs_path = './output/logs'
        if not os.path.exists(self.logs_path):
            os.mkdir(self.logs_path)
        self.save_itr_size = 500

        self.keywords_file = './data/keywords_chinese.dict'
        self.keywords_str = get_keyword_str(self.keywords_file)[1:] + u'卍'
        self.num_class = len(self.keywords_str) + 1
