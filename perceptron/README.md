# 感知机

需要把MNIST手写数字数据集mnist.csv放在目录`../data/`下

## 运行
```bash
python perceptron.py
```

## 可选参数
参数|说明
----|---
LR | 学习率
Batch Size | 一次迭代的批量大小
num_label | 类别数量
num_epoch | 训练全部训练样本的次数
USE_SOFTMAX | 是否使用Softmax计算类别概率
NORMALIZE | 对输入图像进行规范化
USE_HOG | 使用输入图像的HOG特征作为感知机的输入
