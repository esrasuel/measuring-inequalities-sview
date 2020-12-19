#!/bin/sh
data_dir=/rds/general/user/es1510/projects/pathways/live/Transferability/transfer_esra/database/london

## generating partitions
python ordinal_classification_sview.py \
       -m 1 \
       -i ${data_dir}/london_vggcodes.hdf5 \
       -l ${data_dir}/london_labels.csv \
       --label_name dinc_s \
       --clabel_name oa_code \
       --model_name only_train \
       --part_file models/london_onlytrain.partitions \
       --validation_flag 4 \
       --city_name london \
       --num_epochs 50 \
       --batch_size 20 \
       --lrrate 0.0001 \
       --train_part 0.95 \
       --gen_part

## training
python ordinal_classification_sview.py \
       -m 1 \
       -i ${data_dir}/london_vggcodes.hdf5 \
       -l ${data_dir}/london_labels.csv \
       --label_name dinc_s \
       --clabel_name oa_code \
       --model_name only_train \
       --part_file models/london_onlytrain.partitions \
       --validation_flag 4 \
       --city_name london \
       --num_epochs 50 \
       --batch_size 20 \
       --lrrate 0.0001 \
       --train_part 0.95
