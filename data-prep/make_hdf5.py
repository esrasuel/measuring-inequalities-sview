# -*- coding: utf-8 -*-
"""
@author: Esra

Create hdf5 files to be used in training, 

Reads in:
(i) features (codes) extracted from images using VGG16 (extract_features_gview.py)
(ii) metadata containing labels, and input image ids

Outputs [in the directory: ../../learning-ready-data/]
(i) HDF5 file with features (4096D codes extracted from VGG16): [output_name]_images_imgnet.hdf5
(ii) labels file with corresponding variable values [output_name]_variables.p

"""

import pandas as pd
import numpy as np
import h5py
import os
import sys


if len(sys.argv) < 3:
    print ("\n")
    print ("USAGE: python make_hdf5.py <code_name> <output_name>")
    print ("\t code_name: gview_codes_london - filemame for vgg feature npy file")
    print ("\n")
    sys.exit(1)
else:
    narg = len(sys.argv)
    code_name = sys.argv[1]
    output_name = sys.argv[2]
    print("Code name:{}".format(code_name))
    print("Output name:{}".format(output_name))

# this is a pickle file containing img_id, postcode, lsoa codes, and 
#  labels as deciles (for different outcome variables e.g. income, overcrowding etc.) 
#     for each image location 
#  Note: Each img_id will have four images corresponding to four different camera directions

city_pd=pd.read_pickle('../../data/raw-meta-data/ONSPD_AUG_2017_LONDON_W_METADATA_IMGID.p')


cols=city_pd[['img_id', 'pcd', 'oa11', 'lsoa11', 'msoa11'] + list(city_pd.columns[67:])]
lr_cols = cols.drop_duplicates(subset='img_id').reset_index(drop=True)
imgs = list(lr_cols['img_id'])

f = h5py.File('../../data/learning-ready-data/{}_images_imgnet.hdf5'.format(output_name), 'w')
features = f.create_dataset('features', shape=[len(imgs), 4, 4096], dtype='f')
m = 0 # in case there are no images - skip index
for n in range(len(imgs)):
    file_name = '../../data/{}/{}.npz'.format(code_name, imgs[n])
    if os.path.exists(file_name):
        loaded = np.load(file_name)
        features[m, :, :] = loaded['code']
        m = m + 1
    if m % 200 == 0:
        print('Image: {} / {} done.'.format(m, len(imgs)))
        
lr_cols.to_pickle('../../data/learning-ready-data/{}_variables.p'.format(output_name))
