#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 5:13 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : app.py

from config import cnn_ctc_conf
import argparse
import sys
from ModelRunner import ModelRuuner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Define the mode for the process'))
    parser.add_argument(
        '--mode', dest='mode', default='trainval',
        choices={'trainval', 'test'}, required=True,
        help=('What you want?'))
    parser.add_argument(
        '--gpu', dest='gpu', default=0,
        choices={0, 1, -1}, type=int,
        help=('Whether use gpu or not'))

    parser.add_argument(
        '--image_path', dest='image_path', default=None,
        type=str,
        help=('Where the image you want rec?'))
    parameters = parser.parse_args(sys.argv[1:])
    gpu = parameters.gpu
    config = cnn_ctc_conf(gpu=gpu)
    is_training = False
    if parameters.mode == 'trainval':
        is_training = True
    api = ModelRuuner(config, is_training)
    if is_training:
        api.train()
    else:
        pred, score = api.test_file(parameters.image_path)
        print('result: {}\nscore: {}'.format(pred, score))
