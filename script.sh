#!/bin/bash
# Let's call this script venv.sh
source "/home/kasra/Documents/NLI_SFT/.env/bin/activate"


python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Linear \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Iid \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Square \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Exp \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Linear \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Iid \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Square \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Exp \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/

python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Linear \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Iid \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Square \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Exp \
  --epsilon 10 \
  --accountant RDP \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/

python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Linear \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Iid \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Square \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name QNLI \
  --partition_policy Exp \
  --epsilon 10 \
  --accountant Our \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/QNLI/