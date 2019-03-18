import scipy.io
import numpy as np
import os
import scipy.misc

cwd = os.getcwd()
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
vgg = scipy.io.loadmat(VGG_PATH)
# 先显示一下数据类型，发现是dict
print(type(vgg))
# 字典就可以打印出键值dict_keys(['__header__', '__version__', '__globals__', 'layers', 'classes', 'normalization'])
print(vgg.keys())
# 进入layers字段，我们要的权重和偏置参数应该就在这个字段下
layers = vgg['layers']

# 打印下layers发现输出一大堆括号，好复杂的样子：[[ array([[ (array([[ array([[[[ ,顶级array有两个[[
# 所以顶层是两维,每一个维数的元素是array,array内部还有维数
#print(layers)

# 输出一下大小，发现是(1, 43)，说明虽然有两维,但是第一维是”虚的”,也就是只有一个元素
# 根据模型可以知道,这43个元素其实就是对应模型的43层信息(conv1_1,relu,conv1_2…),Vgg-19没有包含Relu和Pool,那么看一层就足以,
# 而且我们现在得到了一个有用的index,那就是layer,layers[layer]
print("layers.shape:", layers.shape)
layer = layers[0]
print("layer.shape:", layer.shape)
conv1_1 = layer[0]
print("conv1_1.shape:", conv1_1.shape)
#print(conv1_1[0][0][0][0][1])
