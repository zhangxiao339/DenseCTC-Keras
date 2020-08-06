#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/6 1:04 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : Trainer.py

from util.DataGenerator import DataGenerator
from model.DenseCTCModel import Dense_CTC_model


def train(config):
    api = Dense_CTC_model(config=config, is_training=True)
    train_data_loader = DataGenerator(data_file=config.train_data_file,
                                      image_path=config.train_image_path,
                                      num_class=config.num_class, keywords=config.keywords_str,
                                      batch_size=config.batch_size,
                                      max_label_length=config.max_label_length,
                                      image_size=(config.image_max_height, config.image_max_width)
                                      )
    test_data_loader = DataGenerator(data_file=config.test_data_file, image_path=config.test_image_path,
                                     num_class=config.num_class, keywords=config.keywords_str,
                                     batch_size=config.batch_size,
                                     max_label_length=config.max_label_length,
                                     image_size=(config.image_max_height, config.image_max_width)
                                     )
    api.train_val(train_data_loader, test_data_loader)