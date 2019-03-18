import numpy as np
import os
import scipy.io
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import tensorflow as tf
#get_ipython().magic(u'matplotlib inline')
cwd = os.getcwd()
print ("Package loaded")
print ("Current folder is %s" % (cwd) )

import os.path
if not os.path.isfile('./data/imagenet-vgg-verydeep-19.mat'):
    get_ipython().system(u'wget -O data/imagenet-vgg-verydeep-19.mat http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')


def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean_pixel = [103.939, 116.779, 123.68]
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel
print ("Network for VGG ready")


# In[4]:

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel
def imread(path):
    return scipy.misc.imread(path).astype(np.float)
#def imsave(path, img):
#    img = np.clip(img, 0, 255).astype(np.uint8)
#    scipy.misc.imsave(path, img)
print ("Functions for VGG ready")

ntrain = 2
imgsize=[64, 64]
cwd  = os.getcwd()
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
IMG_PATH1 = cwd + "/images/cats/images (1).jpeg"
IMG_PATH2 = cwd + "/images/cats/images (2).jpeg"
#IMG_PATH = [cwd + "/images/cat.jpg", cwd + "flash.jpg"]
input_image = np.ndarray((2, 225, 225, 3))
input_image[0, :] = imread(IMG_PATH1)
input_image[1, :] = imread(IMG_PATH2)

#image = imresize(input_image, [imgsize[0],imgsize[1]]])
graysmall = imresize(input_image, [imgsize[0], imgsize[1]])/255.
grayvec   = np.reshape(graysmall, (1, -1))
imgtensor = np.ndarray((1, imgsize[0], imgsize[1], 3))
imgtensor[-1 , :, :, :] = graysmall

#print(input_image.shape)

#input_image = np.reshape(input_image, [imgsize[0], imgsize[1], 3])
#tensor[i, :, :, :] = currimg
with tf.Graph().as_default(), tf.Session() as sess:
    with tf.device("/cpu:0"):
        img_placeholder = tf.placeholder(tf.float32
                                         , shape=(None, imgsize[0], imgsize[1], 3))
        nets, mean_pixel = net(VGG_PATH, img_placeholder)
        features = nets['relu5_4'].eval(feed_dict={img_placeholder: imgtensor})
print("Convolutional map extraction done")

vectorized = np.ndarray((ntrain, 4*4*512))
for i in range(ntrain):
    curr_feat = features[i, :, :, :]
    curr_feat_vec = np.reshape(curr_feat, (1, -1))
    vectorized[i, :] = curr_feat_vec

print("Shape of 'train_vectorized' is %s" % (vectorized.shape,))
print(vectorized.astype(float))

x = tf.placeholder(tf.float32, [None, 4*4*512])
with tf.device("/cpu:0"):
    #n_input  = dim
    #n_output = nclass
    weights  = {
        'wd1': tf.Variable(tf.random_normal([4*4*512, 1024], stddev=0.1)),
    }
    biases   = {
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    }
    def conv_basic(_input, _w, _b):
        # Input
        _input_r = _input
        # Vectorize
        _dense1 = tf.reshape(_input_r, [-1, _w['wd1'].get_shape().as_list()[0]])
        # Fc1
        _fc1 = tf.nn.sigmoid(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        # #_fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # # Fc2
        # _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # # Return everything
        # out = {'input_r': _input_r, 'dense1': _dense1,
        #     'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out }
        return _fc1
    _pred = conv_basic(x, weights, biases)
    init = tf.initialize_all_variables()

print ("Network Ready to Go!")

sess = tf.Session()
sess.run(init)
vec = sess.run(_pred, feed_dict={x: vectorized})
print(vec.shape)
for i in range(vec.shape[1]):
    if vec[:, i] < 0.5:
        vec[:, i] = 0
    else:
        vec[:, i] = 1
print(vec)

