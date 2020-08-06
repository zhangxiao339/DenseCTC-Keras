## <a href="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/README_CN.md">中文</a>
## Introduction
    The basic model of text recognition, based on densenet+ CTC, does not include LSTM, can take into account the recognition rate and performance, CPU on average 5ms on macbookpro i7
    Based on Tensorflow and Keras, the end - to - end detection and recognition of variable length Chinese characters are realized
* text detect：DB
* text rec：DenseNet + CTC

## Environment 
    tensorflow 1.13.1, keras 2.2.4

## demo
<div>
    <img src="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/data/demo/demo.png"/>
    <img src="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/data/demo/demo_result.png"/>
</div>

## train
#### 1. Data
    * 先处理数据集为统一大小
    * 配置config
#### 2. Training
    * python app.py --mode trainval --gpu 'you gpu id'
    * launch tensorboard

## test
* got the text picture
* python app.py --mode test --image_path 'you image file' --gpu 'you gpu id'

## reference
[1] https://github.com/chineseocr/chinese-ocr

[2] https://github.com/YCG09/chinese_ocr