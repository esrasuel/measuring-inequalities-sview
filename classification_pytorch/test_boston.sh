#!/bin/sh
data_dir=/rds/general/user/es1510/projects/pathways/live/Transferability/transfer_esra/database/boston

## training
python ordinal_classification_sview.py \
       -m 0 \
       -i ${data_dir}/boston_vggcodes.hdf5 \
       -l ${data_dir}/boston_labels.csv \
       --label_name dinc_s \
       --clabel_name oa_code \
       --model_name only_train \
       --part_file models/london_onlytrain.partitions \
       --validation_flag 5 \
       --city_name london \
       --num_epochs 50 \
       --batch_size 20 \
       --lrrate 0.0001 \
       --train_part 0.95 \
       --test_city_name boston

