# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#!/usr/bin/env bash


max=3
lr=0.001
lps=3001
det=True

for i in `seq 0 $max`
do
python main_agg_SNR_causality.py \
--lr=$lr  \
--num_classes=7 \
--test_every=100 \
--logs='agg_SNR_causality/logs_'$i'/' \
--batch_size=32 \
--model_path='agg_SNR_causality/models_'$i'/' \
--unseen_index=$i \
--loops_train=$lps \
--step_size=$lps \
--state_dict=$2 \
--data_root=$1 \
--weight_decay=1e-4 \
--momentum=0.9 \
--deterministic=$det
done
