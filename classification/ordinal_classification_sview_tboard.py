#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: esra

"""

from __future__ import division
from __future__ import print_function
#import os.path

import numpy as np
import time as tm
import tensorflow as tf
import sys
import pandas as pd
import scipy.stats as stats
import h5py
from scipy.linalg import toeplitz
import pickle
from sklearn.metrics import confusion_matrix
import network_models as nm
import partitioning
import datasets
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
#print( os.environ['SGE_GPU'])
tf.reset_default_graph()

#================== PARSING INPUTS ==========================
#== TO FINISH THE PARSING AND FEEDING INTO THE MODEL WITH VARIOUS CHOICES ==#
parser = argparse.ArgumentParser(description='ordinal_classification_sview')
parser.add_argument("--mode", "-m", help="Training format >> 0:test, 1:train, 2:fine_tune_test, 3: fine_tune_train", default=1, type=int, choices=[0,1,2,3])
parser.add_argument("--imgfile", "-i", help="hdf5 image file", default=None, type=str)
parser.add_argument("--labfile", "-l", help="label pickle file", default=None, type=str)
parser.add_argument("--labelName", "-n", help="label name e.g. mean-income", default=None, type=str)
parser.add_argument("--clabelName", help="constrain label name for train-test splits e.g. lsoa ", default=None, type=str)
parser.add_argument("--modelName", help="trained model name for saving network weights", default=None, type=str)
parser.add_argument("--gen_part", help="generate or use partition", action="store_true")
parser.add_argument("--part_file", help="partition file", default=None, type=str)
parser.add_argument("--validation_flag", help="cross validation (0) | train-test-validation split (1) | train-test split (2) | train-test per class split (3)", default=1, type=int, choices=[0,1,2,3])
parser.add_argument("--part_kn", help="total number of partitions in cross-validation", default=5, type=int)
parser.add_argument("--part_kp", help="partition number to work with in cross-validation", default=0, type=int)
parser.add_argument("--train_part", help="training set partition size - between 0 and 1", default=0.6, type=float)
parser.add_argument("--test_part", help="test set partition size - between 0 and 1", default=0.3, type=float)
parser.add_argument("--validation_part", help="validation set partition size - between 0 and 1", default=0.1, type=float)
parser.add_argument("--train_size", help="size of the training set to be used - between 0 and 1", default=1.0, type=float)
parser.add_argument("--city_name", help="name of the city", default="london")
parser.add_argument("--num_epochs", help="number of epochs", default=10, type=int)
parser.add_argument("--batch_size", help="batch size", default=10, type=int)
parser.add_argument("--lrrate", help="learning rate", default=5e-6, type=float)
parser.add_argument("--testsetlabel", help="class for test set partition", default=0, type=int)
args = parser.parse_args()

train_format = args.mode # 0: test, 1: train, 2: refine, 3: refine_test,
TRAIN = False
TEST = False
RTRAIN = False
RTEST = False

outformat = 'none'
if train_format == 0:
    TEST = True
    outformat = 'train'
elif train_format == 1:
    TRAIN = True
    outformat = 'train'
elif train_format == 2:
    RTRAIN = True
    outformat = 'refine'
elif train_format == 3:
    RTEST = True
    outformat = 'refine'

img_hdf5_file = args.imgfile
lab_pickle_file = args.labfile
label_name = args.labelName
clabel_name = args.clabelName
trained_model_name = args.modelName
part_gen = args.gen_part # should I generate partitions for mapping
part_file = args.part_file # name of the file of the k partitions (pickle)
validation_flag = args.validation_flag
part_kn = args.part_kn # total number of partitions wanted
part_kp = args.part_kp # the partition number we want to be working with
train_part = args.train_part
test_part = args.test_part
validation_part = args.validation_part
train_size = args.train_size
outformat = '{}_{}'.format(outformat, train_size)
city_name = args.city_name
batch_size=args.batch_size
num_epochs=args.num_epochs
lrrate = args.lrrate
label_test = args.testsetlabel

## print information
if TRAIN: 
    print('Training...')
elif TEST:
    print('Testing...')
elif RTRAIN:
    print('Fine-tuning...')
else:
    print('Testing fine-tuned model...')

print('Image file, Label file: {}, {}'.format(img_hdf5_file, lab_pickle_file))
print('Trained model name: {}'.format(trained_model_name))
print('Label name: {}'.format(label_name))
print('Constrain label name: {}'.format(clabel_name))
print('Generating partitions...{}'.format(part_gen))
print('Partition file name...{}'.format(part_file))
if validation_flag == 0:
    print('Running {} fold cross validation...'.format(part_kn))
elif validation_flag == 1:
    print('Train: {}, Validation: {}, Test: {} divide'.format(train_part, validation_part, test_part))
elif validation_flag == 2:
    print('Train: {}, Test: {} divide'.format(train_part, test_part))
    if train_part + test_part < 1.0:
        print('IMPORTANT: Train and test portions do not add up to 1.')
elif validation_flag == 3:
    print('Test class(es): {}'.format(label_test))
print('Training size...{}'.format(train_size))
print('Final output file name acronym...{}'.format(outformat))
print('City name: {}'.format(city_name))


#==============================================================================
print('loading training dataset...')
# Uses Datasets [datasets.py] to consturct the data for use in training
if validation_flag == 0: # meaning we are doing cross validation
    DS = datasets.Dataset_CrossValidation(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, part_kn=part_kn, part_kp=part_kp, psize=train_size)
elif validation_flag == 1: # meaning we divide the dataset into test / validation / training. 
    DS = datasets.Dataset_TVT(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, train_part, validation_part, psize=train_size)
elif validation_flag == 2: # meaning we divide the dataset into test and train
    DS = datasets.Dataset_TT(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
elif validation_flag == 3: # meaning we divide the dataset into test and train using the class label for test set
    DS = datasets.Dataset_TT_byclass(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name, label_test=label_test)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)

if part_gen:
    sys.exit(1)


print('done.')

#make a simple fully connected network
#==============================================================================

# == define the input place holder == #
# = training data = #
x_inp_1 = tf.placeholder("float", shape=[None, 4096])
x_inp_2 = tf.placeholder("float", shape=[None, 4096])
x_inp_3 = tf.placeholder("float", shape=[None, 4096])
x_inp_4 = tf.placeholder("float", shape=[None, 4096])
labs = tf.placeholder("float", shape=[None, 10])
phase = tf.placeholder(tf.bool, name='phase')

# = validation data = #
x_inp_1_valid = tf.placeholder("float", shape=[None, 4096])
x_inp_2_valid = tf.placeholder("float", shape=[None, 4096])
x_inp_3_valid = tf.placeholder("float", shape=[None, 4096])
x_inp_4_valid = tf.placeholder("float", shape=[None, 4096])
labs_valid = tf.placeholder("float", shape=[None, 10])

# = run the network for training and validation = #
h5 = nm.network((x_inp_1, x_inp_2, x_inp_3, x_inp_4),
                'network', reuse=False, phase=phase)
h5_valid = nm.network((x_inp_1_valid, x_inp_2_valid, x_inp_3_valid, x_inp_4_valid),
                      'network', reuse=True, phase=False)

# = convert to probabilities = #
PC = nm.convert2prob(h5, 'probs')
PC_valid = nm.convert2prob(h5_valid, 'probs_valid')

# == cost function == #
loss_ordinal = tf.reduce_mean(tf.reduce_sum(-tf.multiply(labs,tf.log(PC)), axis=1))
loss_ordinal_valid = tf.reduce_mean(tf.reduce_sum(-tf.multiply(labs_valid,tf.log(PC_valid)), axis=1))

# == training step == #
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(lrrate).minimize(loss_ordinal)

# summaries
tf.summary.scalar('training_loss', loss_ordinal)
tf.summary.scalar('validation_loss', loss_ordinal_valid)
summaries = tf.summary.merge_all()


# start session
#==============================================================================
#==============================================================================
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("Initialized parameters")
saver = tf.train.Saver()
ts = tm.time()


# do the training
#==============================================================================
#==============================================================================
if validation_flag == 0:
    log_file = './log_dirs/logs_{}_{}_{}_{}_out_of_{}_fold'.format(city_name,
                                                                   outformat,
                                                                   trained_model_name,
                                                                   part_kp,
                                                                   part_kn)
    save_name = '../../../../models/classification-ordinal-vmap-rev/{}_{}_{}_{}_out_of_{}_fold'.format(city_name,
                                                                                                       outformat,
                                                                                                       trained_model_name,
                                                                                                       part_kp,
                                                                                                       part_kn)
elif validation_flag == 1:
    log_file = './log_dirs/logs_{}_{}_{}_division_tr{}_vl{}_te{}'.format(city_name,
                                                                         outformat,
                                                                         trained_model_name,
                                                                         train_part,
                                                                         validation_part,
                                                                         test_part)
    save_name = '../../../../models/classification-ordinal-vmap-rev/{}_{}_{}_division_tr{}_vl{}_te{}'.format(city_name,
                                                                                                             outformat,
                                                                                                             trained_model_name,
                                                                                                             train_part,
                                                                                                             validation_part,
                                                                                                             test_part)
elif validation_flag == 2:
    log_file = './log_dirs/logs_{}_{}_{}_division_tr{}_te{}'.format(city_name,
                                                                    outformat,
                                                                    trained_model_name,
                                                                    train_part,
                                                                    test_part)
    save_name = '../../../../models/classification-ordinal-vmap-rev/{}_{}_{}_division_tr{}_te{}'.format(city_name,
                                                                                                        outformat,
                                                                                                        trained_model_name,
                                                                                                        train_part,
                                                                                                        test_part)

elif validation_flag == 3: 
    log_file = './log_dirs/logs_{}_{}_{}_division_classlabel'.format(city_name,
                                                                    outformat,
                                                                    trained_model_name)
    save_name = '../../../../models/classification-ordinal-vmap-rev/{}_{}_{}_division_classlabel'.format(city_name,
                                                                                                        outformat,
                                                                                                        trained_model_name)

    
if TRAIN or RTRAIN:
    #
    iter_per_epoch = np.int(len(DS.train_part) / np.float(batch_size))
    
    if RTRAIN:
        saver.restore(sess, '../../../../models/classification-ordinal-vmap-rev/{}'.format(trained_model_name))
    

    batch_test = DS.get_test_batch(1000)
    batch_train = DS.get_balanced_validation_batch(1000)

    loss_min = 999999

    summary_writer = tf.summary.FileWriter('{}'.format(log_file), graph=tf.get_default_graph())

    with tf.device('/gpu:0'):

        for epoch in range(0, num_epochs):
            for step in range(0, iter_per_epoch):

                batch = DS.get_balanced_train_batch(batch_size)

                _, summaries_, loss_ordinal_valid_ = sess.run([train_step, summaries, loss_ordinal_valid],
                                                              feed_dict={x_inp_1: batch[0], \
                                                                         x_inp_2: batch[1], \
                                                                         x_inp_3: batch[2], \
                                                                         x_inp_4: batch[3], \
                                                                         labs: batch[4], \
                                                                         phase: True, \
                                                                         x_inp_1_valid : batch_test[0], \
                                                                         x_inp_2_valid : batch_test[1], \
                                                                         x_inp_3_valid : batch_test[2], \
                                                                         x_inp_4_valid : batch_test[3], \
                                                                         labs_valid : batch_test[4]})
                summary_writer.add_summary(summaries_, step + epoch*iter_per_epoch)
                if step % 100 == 0:
                    print('step {} / epoch {} done. loss: {}'.format(step, epoch, loss_ordinal_valid_))


                # save the model where we get the lowest mse on the training batch set during training
                if (step > iter_per_epoch*0.90) and (epoch == num_epochs-1): # do it only at the last epoch
                    loss_ordinal_ = sess.run([loss_ordinal],feed_dict={x_inp_1:batch_train[0],
                                                                       x_inp_2:batch_train[1],
                                                                       x_inp_3:batch_train[2],
                                                                       x_inp_4:batch_train[3],
                                                                       labs:batch_train[4],
                                                                       phase:False})[0]
                    if loss_ordinal_ < loss_min:
                        loss_min = loss_ordinal_
                        print('new mse min:{}'.format(loss_min))
                    
                        saver.save(sess, save_name)
                        
    print("elapsed time: {0}".format(tm.time()-ts))
    #
else:
    #
    # == read trained network == #
    saver.restore(sess, save_name)
    print('Restored network ... loading test data')

    # == read test data == #
    test_data = DS.get_test_data()
    print('loaded test data ... predicting')
    # == predict for test data == #
    preds = PC.eval(feed_dict={x_inp_1: test_data[0], x_inp_2: test_data[1],
                                       x_inp_3: test_data[2], x_inp_4: test_data[3],
                                       labs: test_data[4], phase:False})
    h5_vals = h5.eval(feed_dict={x_inp_1: test_data[0], x_inp_2: test_data[1],
                                       x_inp_3: test_data[2], x_inp_4: test_data[3],
                                       labs: test_data[4], phase:False})
    # == write h5 values and predicted classes to file == #
    if validation_flag == 0:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_{}_out_of_{}_folds_h5_vals.csv'.format(city_name,
                                                                                                                    outformat,
                                                                                                                    trained_model_name,
                                                                                                                    part_kp,
                                                                                                                    part_kn)
    elif validation_flag == 1:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_tr{}_vl{}_te{}_h5_vals.csv'.format(city_name,
                                                                                                                         outformat,
                                                                                                                         trained_model_name,
                                                                                                                         train_part,
                                                                                                                         validation_part,
                                                                                                                         test_part)
    elif validation_flag == 2:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_tr{}_te{}_h5_vals.csv'.format(city_name,
                                                                                                                    outformat,
                                                                                                                    trained_model_name,
                                                                                                                    train_part,
                                                                                                                    test_part)
    
    elif validation_flag == 3:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_classlabel_h5_vals.csv'.format(city_name,
                                                                                                            outformat,
                                                                                                            trained_model_name)
    np.savetxt(fname, h5_vals, fmt='%1.5f')
    
    
    print('wrote h5_vals')
    
    
    
    
    if validation_flag == 0:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_{}_out_of_{}_folds_predictions.csv'.format(city_name,
                                                                                                                        outformat,
                                                                                                                        trained_model_name,
                                                                                                                        part_kp,
                                                                                                                        part_kn)
    elif validation_flag == 1:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_tr{}_vl{}_te{}_predictions.csv'.format(city_name,
                                                                                                                             outformat,
                                                                                                                             trained_model_name,
                                                                                                                             train_part,
                                                                                                                             validation_part,
                                                                                                                             test_part)
    elif validation_flag == 2:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_tr{}_te{}_predictions.csv'.format(city_name,
                                                                                                                        outformat,
                                                                                                                        trained_model_name,
                                                                                                                        train_part,
                                                                                                                        test_part)
    elif validation_flag == 3:
        fname = '../../../../analysis/vmap-binomial-predictions-rev/{}_{}_{}_division_classlabel_predictions.csv'.format(city_name,
                                                                                                            outformat,
                                                                                                            trained_model_name)
    DS.write_preds(preds.argmax(axis=1), fname)
    print('wrote predictions')
