# Differential Privacy-Driven Federated Learning for Large Language Models in HMI Systems

The GLUE dataset learning process is using Transformer library and is adopted from https://github.com/nyu-mll/GLUE-baselines <br>
The Federated learning enviroment is using Flower AI framework. https://flowerai.net/docs/framework/index.html

## Experiments
To run the experiments in the paper run:
```
./script.sh
```
## Noise Calculation
To find the proper std deviation of noise in different accountants:
```
Python3 ./noise_calculation/get_noise.py
```
target_epsilons and dataset_size_list is configurable in get_noise.py file.

## Single Experiment
```
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Linear \
  --epsilon 10 \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
```
Model_name is the based model. 
task_name is the dataset which can be (SST2, QNLI, or QQP).
Parition_policy can be (Iid, Linear, Square, or Exp)

