<a href="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/README.md">English</a>
## 简介
    文字识别基础模型，基于densenet+ctc，不包含lstm，可兼顾识别率和性能，cpu上平均耗时5ms on macbookpro i7
    基于Tensorflow和Keras实现端到端的不定长中文字符检测和识别
* 文本检测：DB
* 文本识别：DenseNet + CTC

## 环境部署
    tensorflow 1.13.1, keras 2.2.4

## demo
<div>
    <img src="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/data/demo/demo.png"/>
    <img src="https://github.com/zhangxiao339/DenseCTC-Keras/tree/master/data/demo/demo_result.png"/>
</div>

## 训练
#### 1. 数据准备
    * 先处理数据集为统一大小
    * 配置config
#### 2. 训练
    * 训练
    * 启动tensorboard查看训练情况，调试参数训练提高准确率
    * python app.py --mode trainval --gpu 'you gpu id'

## 测试
* 准备好图像
* python app.py --mode test --image_path 'you image file' --gpu 'you gpu id'

## 参考
[1] https://github.com/chineseocr/chinese-ocr

[2] https://github.com/YCG09/chinese_ocr