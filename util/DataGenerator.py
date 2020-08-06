#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/1 7:19 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : DataGenerator.py
import os
import numpy as np
# from PIL import Image
import cv2


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def loadFile(filename):
    res = []
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1]
    return dic


def build_key_dict(keywords_str):
    result = {}
    idx = 0
    for key in keywords_str:
        result[key] = idx
        idx += 1
    return result

class DataGenerator:
    def __init__(self, data_file, image_path, num_class, keywords, batch_size, max_label_length, image_size=(32, 280)):
        self.data_file = data_file
        self.image_path = image_path
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.image_size = image_size
        self.num_class = num_class
        self.keywords = keywords
        self.key_dict = build_key_dict(self.keywords)
        # assert type in ['train', 'test'], "type error, just support the ['train', 'test']"
        # self.type = type
        self.image_labels = loadFile(data_file)
        self.total_item_size = len(self.image_labels)

    def generate(self):
        image_files = [i for i, j in self.image_labels.items()]
        r_n = random_uniform_num(len(image_files))
        image_files = np.array(image_files)
        while True:
            batch_images = []
            batch_image_lengths = []
            batch_labels = []
            batch_label_lengths = []
            shuff_image_file = image_files[r_n.get(self.batch_size)]
            real_batch_size = 0
            for i, j in enumerate(shuff_image_file):
                image_file = os.path.join(self.image_path, j)
                if not os.path.exists(image_file):
                    print('image not found!')
                    continue
                # img = Image.open(image_file).convert('L')
                img = cv2.imread(image_file, 0)
                h, w = img.shape[:2]
                if h != self.image_size[0] or w != self.image_size[1]:
                    mat = cv2.resize(img, (self.image_size[1], self.image_size[0]))
                else:
                    mat = img
                mat = mat / 255.0 - 0.5
                label_str = self.image_labels[j]
                if len(label_str) <= 0:
                    print("label len < 0", j)
                    continue
                batch_images.append(np.expand_dims(mat, axis=2))
                batch_image_lengths.append(self.image_size[1] // 8)
                batch_label_lengths.append(len(label_str))
                label_padding = np.ones([self.max_label_length]) * self.num_class  # padding here
                # label_padding[:len(label_str)] = [int(k) - 1 for k in label_str]
                label_padding[:len(label_str)] = [int(self.key_dict[k]) for k in list(label_str)]
                batch_labels.append(label_padding)
                real_batch_size += 1
            inputs = {'the_input': np.asarray(batch_images, dtype=np.float),
                      'the_labels': np.asarray(batch_labels),
                      'input_length': np.asarray(batch_image_lengths),
                      'label_length': np.asarray(batch_label_lengths),
                      }
            if real_batch_size != self.batch_size:
                print('got the real data can not equal the set batch size: {} => {}'.format(real_batch_size, self.batch_size))
            outputs = {'ctc': np.zeros([real_batch_size])}
            yield (inputs, outputs)
