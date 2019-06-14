#################################################
## AutoEncoder for img_ring folder
## input folder path:　"./img_ring"
## test image path: "test-6.bmp"
#################################################

import random
import time
import cv2 as cv
import os
## model library
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

input_folder_path = "./img_ring"
test_image_path = "test-6.bmp"
stop_loss = 0.00005
def my_next_batch(num, data):
    '''
    Return a total of `num` random samples.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name)

def read_test_images():
    ## load test image
    dataset_folder = input_folder_path
    print("Test Images loaded")
    num = 0

    for filename in os.listdir(dataset_folder):
        ## read image
        imgpath = os.path.join(dataset_folder,filename)
        img = cv.imread(imgpath,-1)
        img = cv.resize(img, (1224,1024))
        ## from cv.mat to numpy
        img = np.asarray(img)
        img = np.reshape(img/255,[1,img.shape[0]*img.shape[1]])

        ## load
        if(num == 0):
            dataset = img
        else:
            dataset = np.append(dataset,img,axis=0)
        num = num + 1

    return dataset

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')

def max_pool_2x2(x):
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return pool, argmax

def max_unpool_2x2(x, shape):
    inference = tf.image.resize_nearest_neighbor(x, tf.stack([shape[1]*2, shape[2]*2]))
    return inference

def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)

#################################################
# design network architecture
#################################################

## input layer
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape = [None, 1224*1024], name="image_tensor")
x_origin = tf.reshape(x, [-1, 1024, 1224, 1])

## first convolution layer of encoder 
W_e_conv1 = weight_variable([3, 3, 1, 32], "w_e_conv1") ## weight
b_e_conv1 = bias_variable([32], "b_e_conv1") ## bias
h_e_conv1 = tf.nn.sigmoid(tf.add(conv2d(x_origin, W_e_conv1), b_e_conv1)) ## output 
h_e_pool1, argmax_e_pool1 = max_pool_2x2(h_e_conv1)
 
## second convolution layer of encoder 
W_e_conv2 = weight_variable([3, 3, 32, 64], "w_e_conv2") ## weight
b_e_conv2 = bias_variable([64], "b_e_conv2") ## bias
h_e_conv2 = tf.nn.sigmoid(tf.add(conv2d(h_e_pool1, W_e_conv2), b_e_conv2)) ## output 
h_e_pool2, argmax_e_pool2 = max_pool_2x2(h_e_conv2)

## third convolution layer of encoder 80
W_e_conv3 = weight_variable([3, 3, 64, 128], "w_e_conv3") ## weight
b_e_conv3 = bias_variable([128], "b_e_conv3") ## bias
h_e_conv3 = tf.nn.sigmoid(tf.add(conv2d(h_e_pool2, W_e_conv3), b_e_conv3)) ## output 
h_e_pool3, argmax_e_pool2 = max_pool_2x2(h_e_conv3)

## fourth convolution layer of encoder 40
#W_e_conv4 = weight_variable([5, 5, 64, 128], "w_e_conv4") ## weight
#b_e_conv4 = bias_variable([128], "b_e_conv4") ## bias
#h_e_conv4 = tf.nn.relu(tf.add(conv2d(h_e_pool3, W_e_conv4), b_e_conv4)) ## output 
#h_e_pool4, argmax_e_pool2 = max_pool_2x2(h_e_conv4)

code_layer = h_e_pool3
print("code layer shape : %s" % code_layer.get_shape())


## first convolution layer of dncoder
W_d_conv0 = weight_variable([3, 3, 64, 128], "w_d_conv0")  ## weight
b_d_conv0 = bias_variable([128], "b_d_conv0") ## bias
output_shape_d_conv0 = tf.stack([tf.shape(x)[0], 256, 306, 64])
h_d_conv0 = tf.nn.sigmoid(deconv2d(code_layer, W_d_conv0, output_shape_d_conv0)) ## output

## first convolution layer of dncoder
W_d_conv1 = weight_variable([3, 3, 32, 64], "w_d_conv1")  ## weight
b_d_conv1 = bias_variable([64], "b_d_conv1") ## bias
output_shape_d_conv1 = tf.stack([tf.shape(x)[0], 512, 612, 32])
h_d_conv1 = tf.nn.sigmoid(deconv2d(h_d_conv0, W_d_conv1, output_shape_d_conv1)) ## output
 

## second convolution layer of dncoder
W_d_conv2 = weight_variable([3, 3, 1, 32], "w_d_conv2")  ## weight
b_d_conv2 = bias_variable([32], "b_d_conv2") ## bias
output_shape_d_conv2 = tf.stack([tf.shape(x)[0], 1024,1224, 1])
h_d_conv2 = tf.nn.sigmoid(deconv2d(h_d_conv1, W_d_conv2, output_shape_d_conv2)) ## output

x_reconstruct = h_d_conv2
print("reconstruct layer shape : %s" % x_reconstruct.get_shape())

## loss(cost) function
## better loss < 10^-5
cost = tf.reduce_mean(tf.pow(x_reconstruct - x_origin, 2))
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

#################################################
# Create session
#################################################

sess = tf.InteractiveSession()
batch_size = 3
init_op = tf.global_variables_initializer()
sess.run(init_op)
train_data = read_test_images()

#################################################
# Training
#################################################
for epoch in range(20000):
	batch = my_next_batch(batch_size,train_data)
	if epoch < 1500:
		if epoch%100 == 0:
			loss = cost.eval(feed_dict={x:batch})
			print("step %d, loss %g"%(epoch, loss))
	else:
		if epoch%1000 == 0: 
			loss = cost.eval(feed_dict={x:batch})
			print("step %d, loss %g"%(epoch, loss))
	if loss < stop_loss:
		break
	optimizer.run(feed_dict={x: batch})


#################################################
# Save model (untest)
#################################################
x_reconstruct = addNameToTensor(x_reconstruct, "output")

tf.saved_model.simple_save(sess,
            'AE_exported',
            inputs={"image_tensor": x},
            outputs={"output": x_reconstruct})

#################################################
# Testing
################################################# 
## load test image
test_img = cv.imread(test_image_path,0)
test_img = cv.resize(test_img, (1224,1024))
test_img = np.asarray(test_img)
test_img = np.reshape(test_img/255,[1,test_img.shape[0]*test_img.shape[1]])

## 
ori_test_img = test_img
ori_test_img = np.reshape(ori_test_img, (1024, 1224))

## inference
test_img = np.reshape(x_reconstruct.eval(feed_dict={x: test_img}), (1024, 1224))

##
diff_img = np.multiply(np.absolute(test_img - ori_test_img),255)
test_img = np.multiply(test_img,255).astype(np.uint8)
test_img = np.array(test_img)

## output image
cv.imwrite('test-6-1.bmp',test_img)
## difference image
cv.imwrite('test-6-2.bmp',diff_img)

