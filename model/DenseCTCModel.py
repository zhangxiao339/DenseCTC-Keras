#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 5:14 下午
# @Author  : Zhang Xiao
# @Email   : sinceresky@foxmail.com
# @Site    : https://github.com/zhangxiao339
# @File    : DenseCTCModel.py
from keras.layers import Input
from BaodanOCR.component.textRecognizer.DensenetCTC.trainner.model import Densenet
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
import tensorflow as tf
import os
import cv2
import numpy as np


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class Dense_CTC_model(object):
    def __init__(self, config, is_training=False):
        self.config = config
        self.is_training = is_training
        self.is_inited = self.init_model()
        self.graph = tf.get_default_graph()

    def init_model(self):
        try:
            self.input = Input(shape=(self.config.image_max_height, None, 1), name='the_input')
            self.output = Densenet.dense_cnn(self.input, self.config.num_class, self.config.dropout)
            K.set_session(get_session(gpu_fraction=self.config.gpu_frac_mem))
            self.basemodel = Model(inputs=self.input, outputs=self.output)
            model_file = os.path.join(self.config.model_path, self.config.trained_model_file_name)
            if os.path.exists(model_file):
                print("Loading model weights from: {}".format(model_file))
                self.basemodel.load_weights(model_file)
                print('done!')
            else:
                print('can not found the trianed model: {}'.format(model_file))
                if not self.is_training:
                    return False
            if self.is_training:
                self.basemodel.summary()
                labels = Input(name='the_labels', shape=[None], dtype='float32')
                input_length = Input(name='input_length', shape=[1], dtype='int64')
                label_length = Input(name='label_length', shape=[1], dtype='int64')
                loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                    [self.output, labels, input_length, label_length])
                self.train_model = Model(inputs=[self.input, labels, input_length, label_length], outputs=loss_out)
                self.train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=self.config.optimizer,
                                         metrics=['accuracy'])
            return True
        except Exception as e:
            print(e)
            return False

    def decode(self, pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.config.num_class - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(self.config.keywords_str[pred_text[i]])
        score = pred.max(-1).mean()
        return u''.join(char_list), score

    def predict(self, img):
        assert self.is_inited, 'the model not inited!'
        mat = img.copy()
        if mat is None or len(mat.shape) < 2:
            return None, None
        if len(mat.shape) == 3 and mat.shape[2] != 1:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        elif len(mat.shape) == 4:
            mat = cv2.cvtColor(mat, cv2.COLOR_BGRA2GRAY)
        height, width = mat.shape[:2]
        if height != 32:
            width = int((32 / height) * width)
            mat = cv2.resize(mat, (width, 32), interpolation=cv2.INTER_LANCZOS4)
        mat = mat.astype(np.float32) / 255.0 - 0.5
        X = mat.reshape([1, 32, width, 1])
        with self.graph.as_default():
            y_pred = self.basemodel.predict(X)
            y_pred = y_pred[:, :, :]
            out, score = self.decode(y_pred)
            return out, score

    def train_val(self, train_data_loader, test_data_loader):
        assert self.is_inited, 'the model not inited!'
        train_item_size = train_data_loader.total_item_size
        test_item_size = test_data_loader.total_item_size

        checkpoint = ModelCheckpoint(filepath=os.path.join(self.config.model_path, 'weights_densenet-{epoch:02d}-{val_loss:.2f}.h5'),
                                     monitor='val_loss', save_best_only=False, save_weights_only=True)
        lr_schedule = lambda epoch: self.config.learning_rate_init * self.config.learning_rate_decay ** epoch
        learning_rate = np.array([lr_schedule(i) for i in range(self.config.max_epoch_size)])
        changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
        earlystop = EarlyStopping(monitor='val_loss', patience=self.config.patience, verbose=1)
        tensorboard = TensorBoard(log_dir=self.config.logs_path, write_graph=True)

        print('-----------Start training-----------')
        self.train_model.fit_generator(train_data_loader.generate(),
                            steps_per_epoch=train_item_size // self.config.batch_size,
                            epochs=self.config.max_epoch_size,
                            initial_epoch=self.config.init_epoch,
                            validation_data=test_data_loader.generate(),
                            validation_steps=test_item_size // self.config.batch_size,
                            callbacks=[checkpoint, earlystop, changelr, tensorboard])
        print('train done!')
