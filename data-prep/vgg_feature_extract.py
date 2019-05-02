# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:56:34 2017

@author: Esra

Extracting features from pretrained VGG-16 
The outputs result in 4096D vectors for each input image

"""

import numpy as np
from skimage.io import imread
from skimage.transform import rescale as resize
import matplotlib.pylab as plt
import tensorflow as tf
# this is from https://github.com/machrisaa/tensorflow-vgg 
import vgg16
import os.path
import pandas as pd
import tarfile
import glob
import os

def vgg_resizeimg(img):
    im_resized=resize(img,224./min(img.shape[0],img.shape[1])+0.0001)
    if im_resized.shape[0]>im_resized.shape[1]:
        diff=im_resized.shape[0]-224
        im_resized=im_resized[int(diff/2):224+int(diff/2),:,:]
    elif im_resized.shape[1]>im_resized.shape[0]:
        diff=im_resized.shape[1]-224
        im_resized=im_resized[:,int(diff/2):224+int(diff/2),:]
    else:
        pass
    return im_resized

# we have four images per location corresponding to different camera directions
# a = 0 degrees, b = 90 degrees, c = 180 degrees, d = 270 degrees
def convert_single_set(imname, vgg, sess):
    batch = []
    for k in ['a','b','c','d']:
        imgk=imread('{}_{}.png'.format(imname,k))
        batchk = vgg_resizeimg(imgk).reshape((224,224,3))
        batch = batch + [batchk]
    batch = np.asarray(batch)
    prob = sess.run(vgg.fc6, feed_dict = {images : batch})
    return prob

nimgs = 133000 # this is the total number of images you would want to extract features from
# VGG network weights used: pretrained weights from https://github.com/machrisaa/tensorflow-vgg
missing_images = []
with tf.Session() as sess:
    images = tf.placeholder("float", [None, 224, 224,3])
    vgg = vgg16.Vgg16(vgg16_npy_path='../../models/pre-trained-networks/VGG16/vgg16.npy')
    vgg.build(images)
    mod = 0


    for n in range(nimgs):
        if n % 1000 == 0:
            # our input images were stored in .tgz files for every 1000 image
            # we first untar all pngs to a temporary image folder
            # should be changed as necessary
            print("removing all png's from the tmp folder")
            for f in glob.glob('../../data/tmp_images/*.png'):
                os.remove(f)
            file_name = '../../data/images/{}.tgz'.format(np.int(n / 1000))
            print("extracting {}".format(file_name))
            tar = tarfile.open(file_name)
            tar.extractall('../../data/tmp_images/.')
            tar.close()
        # 
        if not os.path.isfile('../../data/gview_codes/{}.npz'.format(n)):
            if os.path.isfile('../../data/tmp_images/{}_a.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_b.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_c.png'.format(n)) and \
               os.path.isfile('../../data/tmp_images/{}_d.png'.format(n)):
                try: 
                    prob = convert_single_set('../../data/tmp_images/{}'.format(n), vgg, sess)
                except:
                    print('conversion of {} did not work...'.format(n))
                    missing_images += [n]
                try:
                    np.savez_compressed('../../data/gview_codes/{}.npz'.format(n), code=prob)
                    print("wrote '../../data/gview_codes/{}.npz'".format(n))
                except:
                    print('saving of {} did not work...'.format(n))
            else:
                missing_images += [n]
        else:
            pass
        if n % 100 == 0:
            print('{}/{} done.'.format(n, nimgs))


