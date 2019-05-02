import tensorflow as tf
import numpy as np

# == define different layers == #
def dense(x, size, scope, reuse=True):
    return tf.contrib.layers.fully_connected(x, size, \
    scope=scope, reuse=reuse, \
    activation_fn=None, \
    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

def dense_relu(x, size, scope, reuse=True):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, \
        scope='dense', reuse=reuse, \
        activation_fn=None, \
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return tf.nn.relu(h1, name='relu')

def dense_BN_relu(x, size, phase_train, scope, reuse=True):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, \
        scope='dense', reuse=reuse, \
        activation_fn=None, \
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, reuse=reuse, is_training=phase_train, scope='bn')
        return tf.nn.relu(h2, name='relu')

# == conversion to probability code == #
def convert2prob(h, scope):
    with tf.variable_scope(scope):
        px = tf.nn.relu(tf.nn.sigmoid(h)-1e-6)
        PC1 = tf.pow(1-px, 9)
        PC2 = PC1 * px * 9./(1-px)
        PC3 = PC2 * px * 8./(2*(1-px))
        PC4 = PC3 * px * 7./(3*(1-px))
        PC5 = PC4 * px * 6./(4*(1-px))
        PC6 = PC5 * px * 5./(5*(1-px))
        PC7 = PC6 * px * 4./(6*(1-px))
        PC8 = PC7 * px * 3./(7*(1-px))
        PC9 = PC8 * px * 2./(8*(1-px))
        PC10 = PC9 * px * 1./(9*(1-px))
        PC = tf.concat([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10], 1)
        return PC

# = 2D convolutional layer = #
def conv2D(x, osize, ksize, scope, padding='SAME', reuse=True, stride=1):
    with tf.variable_scope(scope):
        # osize = output channel size (1d number)
        # ksize = kernal size (2d matrix)
        return tf.contrib.layers.conv2d(x, osize, ksize, \
            activation_fn=None, scope='conv', \
            padding=padding, reuse=reuse, stride=stride,\
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

# = 2D convolutional layer followed by reLU = #
def conv2D_relu(x, osize, ksize, scope, padding='SAME', reuse=True, stride=1):
    with tf.variable_scope(scope):
        # osize = output channel size (1d number)
        # ksize = kernal size (2d matrix)
        h1 = tf.contrib.layers.conv2d(x, osize, ksize, \
            activation_fn=None, scope='conv', \
            padding=padding, reuse=reuse, stride=stride, \
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return tf.nn.relu(h1, name='relu')

def conv2D_BN_relu(x, osize, ksize, phase_train, scope, padding='SAME', reuse=True, stride=1):
    with tf.variable_scope(scope):
        # osize = output channel size (1d number)
        # ksize = kernal size (2d matrix)
        h1 = tf.contrib.layers.conv2d(x, osize, ksize, \
            activation_fn=None, scope='conv', \
            padding=padding, reuse=reuse, stride=stride, \
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, \
                                                reuse=reuse, is_training=phase_train, scope='bn')
        return tf.nn.relu(h2, name='relu')


def max_pool2D(x, ksize, scope, stride=2):
        # ksize == pool size
        with tf.variable_scope(scope):
                return tf.contrib.layers.max_pool2d(x, ksize, stride=stride)

def avg_pool2D(x, ksize, scope, stride=2):
        # ksize == pool size
        with tf.variable_scope(scope):
                return tf.contrib.layers.avg_pool2D(x, ksize, stride=stride)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def conv2D_elu(x, osize, ksize, scope, padding='SAME', reuse=True, stride=1):
    with tf.variable_scope(scope):
        # osize = output channel size (1d number)
        # ksize = kernal size (2d matrix)
        h1 = tf.contrib.layers.conv2d(x, osize, ksize, \
            activation_fn=None, scope='conv', \
            padding=padding, reuse=reuse, stride=stride, \
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return tf.nn.elu(h1, name='elu')

def res_block_solid(x, osize, ksize, scope, padding='SAME', reuse=True, stride=1):
        with tf.variable_scope(scope):
                h1 = tf.contrib.layers.conv2d(x, osize, ksize, \
                   activation_fn=None, scope='conv1', \
                   padding=padding, reuse=reuse, stride=stride, \
                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h1 = tf.nn.relu(h1, name='relu1')
                h2 = tf.contrib.layers.conv2d(h1, osize, ksize, \
                   activation_fn=None, scope='conv2', \
                   padding=padding, reuse=reuse, stride=stride, \
                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                return tf.nn.relu(x + h2, name='relu2')

def res_block_dotted(x, osize, ksize, scope, padding='SAME', reuse=True, stride=2):
        with tf.variable_scope(scope):
                h1 = tf.contrib.layers.conv2d(x, osize, ksize, \
                   activation_fn=None, scope='conv1', \
                   padding=padding, reuse=reuse, stride=stride, \
                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h1 = tf.nn.relu(h1, name='relu1')
                h2 = tf.contrib.layers.conv2d(h1, osize, ksize, \
                   activation_fn=None, scope='conv2', \
                   padding=padding, reuse=reuse, stride=1, \
                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                # approach B: match output dimension using 1x1 convolutions
                hshortcut = tf.contrib.layers.conv2d(x, osize, (1,1), stride=stride, \
                        activation_fn=None, scope='conv_sc', \
                        padding=padding, reuse=reuse, \
                        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                # TBI -- approach A: identity mapping with extra zero entries padded for increasin dimension
                return tf.nn.relu(hshortcut + h2, name='relu2')



# == Network == #
def network(x, scope, reuse=False, phase=True):
    with tf.variable_scope(scope):
        # = first layer = 1024
        h1_1 = dense_BN_relu(x[0], 512, phase, 'layer1', reuse=reuse)
        h1_2 = dense_BN_relu(x[1], 512, phase, 'layer1')
        h1_3 = dense_BN_relu(x[2], 512, phase, 'layer1')
        h1_4 = dense_BN_relu(x[3], 512, phase, 'layer1')
        # = second layer = 512
        h2_1 = dense_BN_relu(h1_1, 256, phase, 'layer2', reuse=reuse)
        h2_2 = dense_BN_relu(h1_2, 256, phase, 'layer2')
        h2_3 = dense_BN_relu(h1_3, 256, phase, 'layer2')
        h2_4 = dense_BN_relu(h1_4, 256, phase, 'layer2')
        # = third layer = 128
        h3_1 = dense_BN_relu(h2_1, 128, phase, 'layer3', reuse=reuse)
        h3_2 = dense_BN_relu(h2_2, 128, phase, 'layer3')
        h3_3 = dense_BN_relu(h2_3, 128, phase, 'layer3')
        h3_4 = dense_BN_relu(h2_4, 128, phase, 'layer3')
        # = aggregation layer = averaging
        h3 = h3_1 / 4. + h3_2 / 4. + h3_3 / 4. + h3_4 / 4.
        # = fourth layer = 64
        h4 = dense_BN_relu(h3, 64, phase, 'layer4', reuse=reuse)
        # = output layer = 1
        h5 = dense(h4, 1, 'layer_output',reuse=reuse)
        return h5


def vgg_network(x, scope, reuse=False, phase=True):
        with tf.variable_scope(scope):
                h1 = conv2D_relu(x, 64, (3,3), 'conv1', padding='SAME', reuse=reuse)
                h1 = conv2D_relu(h1, 64, (3,3), 'conv2', padding='SAME', reuse=reuse)
                p1 = max_pool2D(h1, (2,2), 'p1', stride=2)
                h2 = conv2D_relu(p1, 128, (3,3), 'conv3', padding='SAME', reuse=reuse)
                h2 = conv2D_relu(h2, 128, (3,3), 'conv4', padding='SAME', reuse=reuse)
                p2 = max_pool2D(h2, (2,2), 'p2', stride=2)
                h3 = conv2D_relu(p2, 256, (3,3), 'conv5', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv6', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv7', padding='SAME', reuse=reuse)
                p3 = max_pool2D(h3, (2,2), 'p3', stride=2)
                h4 = conv2D_relu(p3, 512, (3,3), 'conv8', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv9', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv10', padding='SAME', reuse=reuse)
                p4 = max_pool2D(h4, (2,2), 'p4', stride=2)
                h5 = conv2D_relu(p4, 512, (3,3), 'conv11', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv12', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv13', padding='SAME', reuse=reuse)
                p5 = max_pool2D(h5, (2,2), 'p5', stride=2)
                p5 = tf.contrib.layers.flatten(p5)
                f6 = dense_relu(p5, 4096, 'fc1', reuse=reuse)
                f7 = dense_relu(f6, 4096, 'fc2', reuse=reuse)
                f8 = dense(f7,1000,'fc3', reuse=reuse)
                f8 = tf.contrib.layers.softmax(f8)
                return f8

def vgg_modified_network(x, scope, reuse=False, phase=True):
        with tf.variable_scope(scope):
                h1 = conv2D_relu(x, 64, (3,3), 'conv1', padding='SAME', reuse=reuse)
                h1 = conv2D_relu(h1, 64, (3,3), 'conv2', padding='SAME', reuse=reuse)
                p1 = max_pool2D(h1, (2,2), 'p1', stride=2)
                h2 = conv2D_relu(p1, 128, (3,3), 'conv3', padding='SAME', reuse=reuse)
                h2 = conv2D_relu(h2, 128, (3,3), 'conv4', padding='SAME', reuse=reuse)
                p2 = max_pool2D(h2, (2,2), 'p2', stride=2)
                h3 = conv2D_relu(p2, 256, (3,3), 'conv5', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv6', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv7', padding='SAME', reuse=reuse)
                p3 = max_pool2D(h3, (2,2), 'p3', stride=2)
                h4 = conv2D_relu(p3, 512, (3,3), 'conv8', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv9', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv10', padding='SAME', reuse=reuse)
                p4 = max_pool2D(h4, (2,2), 'p4', stride=2)
                h5 = conv2D_relu(p4, 512, (3,3), 'conv11', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv12', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv13', padding='SAME', reuse=reuse)
                p5 = max_pool2D(h5, (2,2), 'p5', stride=2)
                p5 = tf.contrib.layers.flatten(p5)
                f6 = dense_relu(p5, 4096, 'fc1', reuse=reuse)
                #f7 = dense_relu(f6, 4096, 'fc2', reuse=reuse)
                #f8 = dense(f7,1000,'fc3', reuse=reuse)
                #f8 = tf.contrib.layers.softmax(f8)
                return f6

def network_vgg_fc(x, scope, reuse=False, phase=True):
        with tf.variable_scope(scope):
                # = vgg layers = 4096 is output size
                v1_1 = vgg_modified_network(x[0], 'vgg', reuse = reuse) 
                v1_2 = vgg_modified_network(x[1], 'vgg', reuse = True)
                v1_3 = vgg_modified_network(x[2], 'vgg', reuse = True)
                v1_4 = vgg_modified_network(x[3], 'vgg', reuse = True)

                # = first layer = 1024
                h1_1 = dense_BN_relu(v1_1, 512, phase, 'layer1', reuse=reuse)
                h1_2 = dense_BN_relu(v1_2, 512, phase, 'layer1')
                h1_3 = dense_BN_relu(v1_3, 512, phase, 'layer1')
                h1_4 = dense_BN_relu(v1_4, 512, phase, 'layer1')
                # = second layer = 512
                h2_1 = dense_BN_relu(h1_1, 256, phase, 'layer2', reuse=reuse)
                h2_2 = dense_BN_relu(h1_2, 256, phase, 'layer2')
                h2_3 = dense_BN_relu(h1_3, 256, phase, 'layer2')
                h2_4 = dense_BN_relu(h1_4, 256, phase, 'layer2')
                # = third layer = 128
                h3_1 = dense_BN_relu(h2_1, 128, phase, 'layer3', reuse=reuse)
                h3_2 = dense_BN_relu(h2_2, 128, phase, 'layer3')
                h3_3 = dense_BN_relu(h2_3, 128, phase, 'layer3')
                h3_4 = dense_BN_relu(h2_4, 128, phase, 'layer3')
                # = aggregation layer = averaging
                h3 = h3_1 / 4. + h3_2 / 4. + h3_3 / 4. + h3_4 / 4.
                # = fourth layer = 64
                h4 = dense_BN_relu(h3, 64, phase, 'layer4', reuse=reuse)
                # = output layer = 1
                h5 = dense(h4, 1, 'layer_output',reuse=reuse)
                return h5



def resnet_network(x, scope, reuse=False, phase=True):
        with tf.variable_scope(scope):
                h1 = conv2D_relu(x, 64, (7,7), 'conv1', padding='SAME', reuse=reuse, stride=2)
                p1 = max_pool2D(h1, (2,2), 'p1', stride=2)
                h2 = res_block_solid(p1, 64, (3,3), 'res1', padding='SAME', reuse=reuse)
                h3 = res_block_solid(h2, 64, (3,3), 'res2', padding='SAME', reuse=reuse)
                h4 = res_block_solid(h3, 64, (3,3), 'res3', padding='SAME', reuse=reuse)
                h5 = res_block_dotted(h4, 128, (3,3), 'res4', padding='SAME', reuse=reuse)
                h6 = res_block_solid(h5, 128, (3,3), 'res5', padding='SAME', reuse=reuse)
                h7 = res_block_solid(h6, 128, (3,3), 'res6', padding='SAME', reuse=reuse)
                h8 = res_block_solid(h7, 128, (3,3), 'res7', padding='SAME', reuse=reuse)
                h9 = res_block_dotted(h8, 512, (3,3), 'res8', padding='SAME', reuse=reuse)
                h10 = res_block_solid(h9, 512, (3,3), 'res9', padding='SAME', reuse=reuse)
                h11 = res_block_solid(h10, 512, (3,3), 'res10', padding='SAME', reuse=reuse)
                p2 = global_avg_pooling(h11)
                p2 = tf.contrib.layers.flatten(p2)
                f1 = dense(p2,1000,'fc3', reuse=reuse)
                f1 = tf.contrib.layers.softmax(f1)                
                return f1



def cnn_network(x, scope, reuse=False, phase=True):
        h1_1 = vgg_network(x[0], 'vgg1')
        h1_2 = vgg_network(x[1], 'vgg1', reuse=True)
        h1_3 = vgg_network(x[2], 'vgg1', reuse=True)
        h1_4 = vgg_network(x[3], 'vgg1', reuse=True)
        # = aggregation layer = averaging
        h2 = h1_1 / 4. + h1_2 / 4. + h1_3 / 4. + h1_4 / 4.
        # = fourth layer = 64
        h4 = dense_BN_relu(h3, 64, phase, 'layer4', reuse=reuse)
        # = output layer = 1
        h5 = dense(h4, 1, 'layer_output',reuse=reuse)
        return h5


def vgg_network(x, scope, reuse=False, phase=True):
        with tf.variable_scope(scope):
                h1 = conv2D_relu(x, 64, (3,3), 'conv1', padding='SAME', reuse=reuse)
                h1 = conv2D_relu(h1, 64, (3,3), 'conv2', padding='SAME', reuse=reuse)
                p1 = max_pool2D(h1, (2,2), 'p1', stride=2)
                h2 = conv2D_relu(p1, 128, (3,3), 'conv3', padding='SAME', reuse=reuse)
                h2 = conv2D_relu(h2, 128, (3,3), 'conv4', padding='SAME', reuse=reuse)
                p2 = max_pool2D(h2, (2,2), 'p2', stride=2)
                h3 = conv2D_relu(p2, 256, (3,3), 'conv5', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv6', padding='SAME', reuse=reuse)
                h3 = conv2D_relu(h3, 256, (3,3), 'conv7', padding='SAME', reuse=reuse)
                p3 = max_pool2D(h3, (2,2), 'p3', stride=2)
                h4 = conv2D_relu(p3, 512, (3,3), 'conv8', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv9', padding='SAME', reuse=reuse)
                h4 = conv2D_relu(h4, 512, (3,3), 'conv10', padding='SAME', reuse=reuse)
                p4 = max_pool2D(h4, (2,2), 'p4', stride=2)
                h5 = conv2D_relu(p4, 512, (3,3), 'conv11', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv12', padding='SAME', reuse=reuse)
                h5 = conv2D_relu(h5, 512, (3,3), 'conv13', padding='SAME', reuse=reuse)
                p5 = max_pool2D(h5, (2,2), 'p5', stride=2)
                p5 = tf.contrib.layers.flatten(p5)
                f6 = dense_relu(p5, 4096, 'fc1', reuse=reuse)
                f7 = dense_relu(f6, 4096, 'fc2', reuse=reuse)
                f8 = dense(f7,1000,'fc3', reuse=reuse)
                f8 = tf.contrib.layers.softmax(f8)
                return f8

