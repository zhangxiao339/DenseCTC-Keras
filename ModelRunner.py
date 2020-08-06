#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/6 1:07 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : ModelRunner.py
from model.DenseCTCModel import Dense_CTC_model
from Trainer import train
import os
import cv2


class ModelRuuner:
    def __init__(self, config, is_training=False):
        self.api = Dense_CTC_model(config=config, is_training=is_training)
        self.config = config

    def test_file(self, image_file):
        assert os.path.exists(image_file), 'input image not found!'
        img = cv2.imread(image_file, 0)
        pred, score = self.api.predict(img)
        return pred, score

    def train(self):
        train(self.config)