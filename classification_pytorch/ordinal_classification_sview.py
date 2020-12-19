from __future__ import division
from __future__ import print_function

# all libraries:
import numpy as np
import time as tm
import sys

import pandas as pd
import scipy.stats as stats
import h5py
from scipy.linalg import toeplitz
import pickle
import argparse
import os
from skimage.io import imsave

# local libraries:
sys.path.insert(1, './tools')
import partitioning
import datasets_sview as datasets

# deep learning part: 
import torch 
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#================== PARSING INPUTS ==========================
#== TO FINISH THE PARSING AND FEEDING INTO THE MODEL WITH VARIOUS CHOICES ==#
parser = argparse.ArgumentParser(description='ordinal_classification_sview')
parser.add_argument("--mode", "-m", help="Training format >> 0:test, 1:train, 2:fine_tune_test, 3: fine_tune_train", default=1, type=int, choices=[0,1,2,3])
parser.add_argument("--img_file", "-i", help="hdf5 image file", default=None, type=str)
parser.add_argument("--lab_file", "-l", help="label pickle file", default=None, type=str)
parser.add_argument("--label_name", "-n", help="label name", default=None, type=str)
parser.add_argument("--clabel_name", help="constrain label name", default=None, type=str)
parser.add_argument("--model_name", help="trained model name", default=None, type=str)
parser.add_argument("--gen_part", help="generate or use partition", action="store_true")
parser.add_argument("--part_file", help="partition file", default=None, type=str)
parser.add_argument("--validation_flag", help="cross validation (0) | " + \
    "train-test-validation split (1) | " + \
    "train-test split (2) | " + \
    "train-test per class split (3)" + \
    "train only split (4)" + \
    "test-only split (5)", default=1, type=int, choices=[0,1,2,3,4,5])
parser.add_argument("--part_kn", help="total number of partitions in cross-validation", default=5, type=int)
parser.add_argument("--part_kp", help="partition number to work with in cross-validation", default=0, type=int)
parser.add_argument("--train_part", help="training set partition size - between 0 and 1", default=0.6, type=float)
parser.add_argument("--test_part", help="test set partition size - between 0 and 1", default=0.3, type=float)
parser.add_argument("--validation_part", help="validation set partition size - between 0 and 1", default=0.1, type=float)
parser.add_argument("--train_size", help="size of the training set to be used - between 0 and 1", default=1.0, type=float)
parser.add_argument("--city_name", help="name of the city used for training", default="london")
parser.add_argument("--num_epochs", help="number of epochs", default=10, type=int)
parser.add_argument("--batch_size", help="batch size", default=10, type=int)
parser.add_argument("--lrrate", help="learning rate", default=2e-6, type=float)
parser.add_argument("--testsetlabel", help="class for test set partition", default=0, type=int)
parser.add_argument("--gpu_num", help="number of the gpu: 0,1,2", default="0", type=str)
parser.add_argument("--aug_prob", help="augmentation probability", default=0.0, type=float)
parser.add_argument("--test_city_name", help="name of the city used for testing - only use with validation_flag=5", default="london", type=str)
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


img_hdf5_file = args.img_file
lab_pickle_file = args.lab_file
label_name = args.label_name
clabel_name = args.clabel_name
trained_model_name = args.model_name
part_gen = args.gen_part # should I generate partitions for mapping
part_file = args.part_file # name of the file of the k partitions (pickle)
validation_flag = args.validation_flag
part_kn = args.part_kn # total number of partitions wanted
part_kp = args.part_kp # the partition number we want to be working with
train_part = args.train_part
test_part = args.test_part
validation_part = args.validation_part
train_size = args.train_size
city_name = args.city_name
batch_size=args.batch_size
num_epochs=args.num_epochs
lrrate = args.lrrate
label_test = args.testsetlabel
aug_prob = args.aug_prob
# = name that will be used to write the model from now on = #
rds_path = './{}/'.format(city_name)
if not os.path.exists(rds_path):
    os.makedirs(rds_path)
    os.makedirs(rds_path + 'log_dirs')
    os.makedirs(rds_path + 'models')
    os.makedirs(rds_path + 'analysis')
part_file = rds_path + part_file
out_pre = '{}_{}_{}'.format(city_name, outformat, trained_model_name)

# = setting the GPU to use = #
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

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
    print('Running {} fold cross validation with validation part size {}'.format(part_kn, validation_part))
elif validation_flag == 1:
    print('Train: {}, Validation: {}, Test: {} divide'.format(train_part, validation_part, test_part))
elif validation_flag == 2:
    print('Train: {}, Test: {} divide'.format(train_part, test_part))
    if train_part + test_part < 1.0:
        print('IMPORTANT: Train and test portions do not add up to 1.')
elif validation_flag == 3:
    print('Test class(es): {}'.format(label_test))
elif validation_flag == 4: 
    validation_part = 1.0 - train_part
    print('Train: {}, Validation:{} divide'.format(train_part, validation_part))
elif validation_flag == 5: 
    print('Test only run. All samples will be used to test the trained model.')
    test_city_name = args.test_city_name
else: 
    print('Validation flag: {} is not supported'.format(validation_flag))

print('Training size...{}'.format(train_size))
print('Final output file name acronym...{}'.format(outformat))
print('City name: {}'.format(city_name))


#==============================================================================
print('loading training dataset...')
if validation_flag == 0: # meaning we are doing cross validation
    DS = datasets.Dataset_CrossValidation(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, part_kn=part_kn, part_kp=part_kp, vsize=validation_part)
elif validation_flag == 1: # meaning we divide the dataset into test / validation / training.
    DS = datasets.Dataset_TVT(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, train_part, validation_part, psize=train_size)
elif validation_flag == 2: # meaning we divide the dataset into test and train
    DS = datasets.Dataset_TT(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
elif validation_flag == 3: # meaning we divide the dataset into test and train using the class label for test set
    DS = datasets.Dataset_TT_byclass(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name, label_test=label_test)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
elif validation_flag == 4: 
    DS = datasets.Dataset_OnlyTrain(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label(part_gen, part_file, train_part, psize=train_size)
elif validation_flag == 5: 
    DS = datasets.Dataset_OnlyTest(img_hdf5_file, lab_pickle_file, label_name, clabel_name=clabel_name)
    DS.pick_label()
if part_gen:
    sys.exit(1)

print('done.')

## Need to generate the U-Net with pytorch. 
# convert the network output to probabilities

def convert_labels(labels):
    newlabels = np.concatenate((np.sum(labels[:,0:1], axis=1)[:,None], 
                                np.sum(labels[:,1:9], axis=1)[:,None],
                                np.sum(labels[:,9:10],axis=1)[:,None]),axis=1)
    return labels, newlabels

def convert2log_prob(h):
    logpx = -torch.log(1 + torch.exp(-h))
    lognx = -h - torch.log(1 + torch.exp(-h))
    logPC1 = 9*lognx
    logPC2 = logPC1 + np.log(9.0) + logpx - lognx
    logPC3 = logPC2 + np.log(8.0/2.0) + logpx - lognx
    logPC4 = logPC3 + np.log(7.0/3.0) + logpx - lognx
    logPC5 = logPC4 + np.log(6.0/4.0) + logpx - lognx
    logPC10 = 9*logpx
    logPC9 = logPC10 + np.log(9.0) + lognx - logpx
    logPC8 = logPC9 + np.log(8.0/2.0) + lognx - logpx
    logPC7 = logPC8 + np.log(7.0/3.0) + lognx - logpx
    logPC6 = logPC7 + np.log(6.0/4.0) + lognx - logpx
    logPC = torch.cat([logPC1, logPC2, logPC3, logPC4, logPC5, logPC6, logPC7, logPC8, logPC9, logPC10], dim=1)
    return logPC

class Net(nn.Module):
    def __init__(self, n_in, n_out): 
        super(Net, self).__init__()
        self.dense1 = nn.Linear(n_in, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, n_out)

    def forward(self, x1, x2, x3, x4): 
        # first layer
        x1 = F.relu(self.dense1(x1))
        x2 = F.relu(self.dense1(x2))
        x3 = F.relu(self.dense1(x3))
        x4 = F.relu(self.dense1(x4))
        # second layer
        x1 = F.relu(self.dense2(x1))
        x2 = F.relu(self.dense2(x2))
        x3 = F.relu(self.dense2(x3))
        x4 = F.relu(self.dense2(x4))
        # third layer
        x1 = F.relu(self.dense3(x1))
        x2 = F.relu(self.dense3(x2))
        x3 = F.relu(self.dense3(x3))
        x4 = F.relu(self.dense3(x4))
        # = aggregation layer = average = #
        x = x1 / 4.0 + x2 / 4.0 + x3 / 4.0 + x4 / 4.0
        # fourth layer
        x = F.relu(self.dense4(x))
        # out layer
        out = self.dense5(x)
        return out, convert2log_prob(out)

net = Net(4096, 1)
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=lrrate, weight_decay=0.0001)

# do the training
#==============================================================================
#==============================================================================
# make it multiple of 200
if validation_flag == 0:
    log_file = rds_path + 'log_dirs/logs_{}_{}_out_of_{}_fold'.format(out_pre, part_kp, part_kn)
    save_name = rds_path + 'models/{}_{}_out_of_{}_fold'.format(out_pre, part_kp, part_kn)
elif validation_flag == 1:
    log_file = rds_path + 'log_dirs/logs_{}_division_tr{}_vl{}_te{}'.format(out_pre,
                                                                   train_part,
                                                                   validation_part,
                                                                   test_part)
    save_name = rds_path + 'models/{}_division_tr{}_vl{}_te{}'.format(out_pre,
                                                              train_part,
                                                              validation_part,
                                                              test_part)
elif validation_flag == 2:
    log_file = rds_path + 'log_dirs/logs_{}_division_tr{}_te{}'.format(out_pre, train_part, test_part)
    save_name = rds_path + 'models/{}_division_tr{}_te{}'.format(out_pre, train_part, test_part)

elif validation_flag == 3:
    log_file = rds_path + 'log_dirs/logs_{}_division_classlabel'.format(out_pre)
    save_name = rds_path + 'models/{}_division_classlabel'.format(out_pre)

elif validation_flag in [4,5]: 
    log_file = rds_path + 'log_dirs/logs_{}_onlytrain'.format(out_pre)
    save_name = rds_path + 'models/{}_onlytrain'.format(out_pre)

if TRAIN or RTRAIN:
    #
    iter_per_epoch = np.int(len(DS.train_part) / np.float(batch_size))

    if RTRAIN:
        print('not yet implemented...')
        sys.exit()

    loss_min = 999999
    # compute the MAE error before training...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on: ', device)

    # computing validation error before training
    mae_error = 0
    cxloss = 0
    runi = 0
    cla_error = 0
    for i, valid_data in enumerate(DS.validation_iterator(batch_size)): 
        xsvi1 = torch.from_numpy(valid_data[0]).float().to(device)
        xsvi2 = torch.from_numpy(valid_data[1]).float().to(device)
        xsvi3 = torch.from_numpy(valid_data[2]).float().to(device)
        xsvi4 = torch.from_numpy(valid_data[3]).float().to(device)
        converted_labels = convert_labels(valid_data[4])[0]
        preds_logits, log_preds_probs = net(xsvi1, xsvi2, xsvi3, xsvi4)
        preds_logits = preds_logits.detach().cpu().numpy()
        log_preds_probs = log_preds_probs.detach().cpu().numpy()
        cxloss += np.mean(-np.sum(converted_labels * log_preds_probs, axis=1))
        mae_error += np.mean(np.abs(np.argmax(log_preds_probs, axis=1) -
                                    np.argmax(converted_labels, axis=1)))
        cla_error += np.mean(np.argmax(log_preds_probs, axis=1) !=
                                 np.argmax(converted_labels, axis=1))
        runi += 1
    mae_error = mae_error / np.float(runi)
    cxloss = cxloss / np.float(runi)
    cla_error = cla_error / np.float(runi)
    print('>>> MAE in validation before training: {0:1.4f}'.format(mae_error))
    print('>>> Xentropy in validation before training: {0:1.4f}'.format(cxloss))
    print('>>> Classification error in validation before training: {0:1.4f}'.format(cla_error))
    loss_min = mae_error
    mae_error_list = [mae_error]
    cxloss_list = [cxloss]
    cla_error_list = [cla_error]
    running_loss = 0.0
    
    for epoch in range(0,num_epochs): 
        optimizer.zero_grad()
        running_loss = 0.0
        for step in range(iter_per_epoch):
            batch = DS.get_balanced_train_batch(batch_size)
            xsvi1 = torch.from_numpy(batch[0]).float().to(device)
            xsvi2 = torch.from_numpy(batch[1]).float().to(device)
            xsvi3 = torch.from_numpy(batch[2]).float().to(device)
            xsvi4 = torch.from_numpy(batch[3]).float().to(device)
            converted_labels = convert_labels(batch[4])[0]
            converted_labels = torch.from_numpy(converted_labels).float().to(device)
            log_preds = net(xsvi1, xsvi2, xsvi3, xsvi4)[1]
            loss = -1.0*torch.mean(torch.sum(converted_labels * log_preds, 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss = running_loss + loss.detach().cpu().numpy()
            if step % 200 == 199:
                print('training loss: {0:1.4f} at iteration {1}'.format(running_loss / 200.0, step+1))
                running_loss = 0.0
                
        mae_error = 0
        cxloss = 0
        runi = 0
        cla_error = 0
        for i, valid_data in enumerate(DS.validation_iterator(batch_size)): 
            xsvi1 = torch.from_numpy(valid_data[0]).float().to(device)
            xsvi2 = torch.from_numpy(valid_data[1]).float().to(device)
            xsvi3 = torch.from_numpy(valid_data[2]).float().to(device)
            xsvi4 = torch.from_numpy(valid_data[3]).float().to(device)
            converted_labels = convert_labels(valid_data[4])[0]
            preds_logits, log_preds_probs = net(xsvi1, xsvi2, xsvi3, xsvi4)
            preds_logits = preds_logits.detach().cpu().numpy()
            log_preds_probs = log_preds_probs.detach().cpu().numpy()
            cxloss += np.mean(-np.sum(converted_labels * log_preds_probs, axis=1))
            mae_error += np.mean(np.abs(np.argmax(log_preds_probs, axis=1) -
                                        np.argmax(converted_labels, axis=1)))
            cla_error += np.mean(np.argmax(log_preds_probs, axis=1) !=
                                 np.argmax(converted_labels, axis=1))
            runi += 1
        mae_error = mae_error / np.float(runi)
        cxloss = cxloss / np.float(runi)
        cla_error = cla_error / np.float(runi)
        print('>>> MAE in validation after epoch {} : {}'.format(epoch + 1, mae_error))
        print('>>> Xentropy in validation after epoch {} : {}'.format(epoch + 1, cxloss))
        print('>>> Classification error in validation after epoch {} : {}'.format(epoch + 1, cla_error))
        mae_error_list.append(mae_error)
        cxloss_list.append(cxloss)
        cla_error_list.append(cla_error)
        # writing if the current iteration reaches a lower validation error: as average of the last 5, to avoid lucky drops.
        if epoch >= 0: 
            if np.mean(mae_error_list[-1]) < loss_min:
                loss_min = np.mean(mae_error_list[-1])
                print('*** Got a new minimum validation loss: {} ***', loss_min)
                torch.save(net.state_dict(), save_name + '_sview_network.pth')



    # writing down the mae error in the validation set over the epochs
    mae_error_name = rds_path + 'analysis/{}_mae_error_list.txt'.format(out_pre)
    np.savetxt(mae_error_name, mae_error_list)
    xent_error_name = rds_path + 'analysis/{}_xentropy_error_list.txt'.format(out_pre)
    np.savetxt(xent_error_name, cxloss_list)
    cla_error_name = rds_path + 'analysis/{}_classification_error_list.txt'.format(out_pre)
    np.savetxt(cla_error_name, cla_error_list)

else:
    print('Loading: ', save_name + '_sview_network.pth')
    net.load_state_dict(torch.load(save_name + '_sview_network.pth'))
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Restored network... running through the test set')
    # == assigning the file names == #
    # == assigning the file names == #
    if validation_flag == 0:
        fname_h5 = rds_path + 'analysis/{}_{}_out_of_{}_folds_h5_vals'.format(out_pre,part_kp, part_kn)
        fname_pred = rds_path + 'analysis/{}_{}_out_of_{}_folds_predictions'.format(out_pre, part_kp, part_kn)
    elif validation_flag == 1:
        fname_h5 = rds_path + 'analysis/{}_division_tr{}_vl{}_te{}_h5_vals'.format(out_pre, train_part, validation_part, test_part)
        fname_pred = rds_path + 'analysis/{}_division_tr{}_vl{}_te{}_predictions'.format(out_pre, train_part, validation_part, test_part)
    elif validation_flag == 2:
        fname_h5 = rds_path + 'analysis/{}_division_tr{}_te{}_h5_vals'.format(out_pre, train_part, test_part)
        fname_pred = rds_path + 'analysis/{}_division_tr{}_te{}_predictions'.format(out_pre, train_part, test_part)
    elif validation_flag == 3:
        fname_h5 = rds_path + 'analysis/{}_division_classlabel_h5_vals'.format(out_pre)
        fname_pred = rds_path + 'analysis/{}_division_classlabel_predictions'.format(out_pre)
    elif validation_flag == 5: 
        fname_h5 = rds_path + 'analysis/{}_testing_on_city_{}_h5vals'.format(out_pre, test_city_name)
        fname_pred = rds_path + 'analysis/{}_testing_on_city_{}_predictions'.format(out_pre, test_city_name)

    # computing validation error before training
    mae = 0
    total_test_num = len(DS.test_part)
    preds_list = []
    h5_list = []
    vmap_list = []
    runi = 0
    for i, test_data_ in enumerate(DS.test_iterator(batch_size)): 
        test_data = test_data_[0]
        vmap = test_data_[1]
        xsvi1 = torch.from_numpy(test_data[0]).float().to(device)
        xsvi2 = torch.from_numpy(test_data[1]).float().to(device)
        xsvi3 = torch.from_numpy(test_data[2]).float().to(device)
        xsvi4 = torch.from_numpy(test_data[3]).float().to(device)
        labels = torch.from_numpy(convert_labels(test_data[4])[0]).float().to(device)
        preds_logits, log_preds_probs = net(xsvi1, xsvi2, xsvi3, xsvi4)
        preds_ = log_preds_probs.to('cpu').detach().numpy()
        h5_ = preds_logits.to('cpu').detach().numpy()
        mae += np.sum(np.abs(np.argmax(preds_, axis=1) - np.argmax(test_data[4], axis=1)))
        preds_list.append(preds_)
        h5_list.append(h5_)
        vmap_list.append(np.concatenate([vmap, h5_, np.argmax(preds_, axis=1)[:,None], np.argmax(test_data[4], axis=1)[:,None]], axis=1))
        runi += batch_size
        if i % 500 == 499:
            print('MAE error is {} with {} samples'.format(mae / runi, runi))
    
    print('Test data MAE: {}'.format(mae / np.float(total_test_num)))
    
    df = pd.DataFrame(np.concatenate(vmap_list, axis=0), columns=['new_imgid', 'gsv_lat', 'gsv_lng', 'oa_code', 'h5', 'predicted', DS.label_name])
    df.to_csv(fname_pred + '.csv', index=False) 
    print('wrote down predictions.')
    

